"""tools/evaluate.py - 동영상 파일로 카운터 정확도를 평가/튜닝하는 도구.

사용 예
  python tools/evaluate.py video.mp4                       # 점프 수와 시각 출력
  python tools/evaluate.py video.mp4 --gt truth.kva        # 정답(Kinovea .kva/.html)과 비교
  python tools/evaluate.py video.mp4 --gt truth.kva --sweep  # threshold 여러 값 비교
  python tools/evaluate.py a.mp4 b.mp4 --gt a.kva b.kva --model lite full

정답 파일: Kinovea 키프레임(.kva)의 UserTime 또는 .html 표의 시각 목록.
포즈 추론 결과는 --cache-dir 에 저장해 두므로 같은 영상을 다른 파라미터로
다시 평가할 때는 몇 초면 끝난다.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jump_counter import JumpCounter, Landmark, extract_signal  # noqa: E402


# --------------------------------------------------------------------- 정답
def load_ground_truth(path: str) -> list[float]:
    p = Path(path)
    if p.suffix.lower() == ".kva":
        root = ET.parse(p).getroot()
        times = [float(pos.get("UserTime")) for pos in root.iter("Position") if pos.get("UserTime")]
    else:
        txt = re.sub(r"<[^>]+>", " ", p.read_text(encoding="utf-8-sig", errors="replace"))
        times = sorted({float(x) for x in re.findall(r"\b\d+\.\d+\b", txt)})
    return sorted(times)


def match_events(pred: list[float], gt: list[float], tol: float = 0.35):
    """예측 시각과 정답 시각을 tol 초 안에서 1:1로 짝짓는다. (tp, fp, fn, 시간차 목록)"""
    used = [False] * len(gt)
    tp = 0
    offsets = []
    for t in pred:
        best, best_d = None, tol
        for j, g in enumerate(gt):
            if not used[j] and abs(g - t) <= best_d:
                best, best_d = j, abs(g - t)
        if best is not None:
            used[best] = True
            tp += 1
            offsets.append(t - gt[best])
    return tp, len(pred) - tp, len(gt) - tp, offsets


# --------------------------------------------------------------------- 추론
def _cache_key(video: Path, model: str, max_side: int) -> str:
    st = video.stat()
    raw = f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}|{model}|{max_side}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def extract_series(video: Path, model: str, cache_dir: Path, max_side: int = 0,
                   verbose: bool = True) -> dict:
    """영상 전체를 포즈 추론해 프레임별 (t, 랜드마크 요약) 목록을 만든다. 캐시 사용."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{video.stem}_{model}_{_cache_key(video, model, max_side)}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    import cv2

    from pose_backend import create_landmarker, detect_landmarks  # noqa: E402

    landmarker = create_landmarker(model)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0
    t_start = time.perf_counter()
    infer_ms = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if max_side and max(h, w) > max_side:
            s = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        res = detect_landmarks(landmarker, frame, int(idx * 1000 / fps))
        infer_ms.append((time.perf_counter() - t0) * 1000)
        lms = None
        if res:
            lms = [(l.x, l.y, 1.0 if l.visibility is None else float(l.visibility)) for l in res]
        frames.append((idx / fps, lms))
        idx += 1
        if verbose and idx % 100 == 0:
            print(f"  {video.name}: {idx}/{total} 프레임 ({time.perf_counter() - t_start:.1f}s)", end="\r")
    cap.release()
    landmarker.close()
    if verbose:
        print(f"  {video.name}: {idx} 프레임, 추론 평균 {statistics.mean(infer_ms):.1f} ms ({model})      ")
    series = {"video": str(video), "model": model, "fps": fps, "width": w, "height": h,
              "frames": frames, "infer_ms": statistics.mean(infer_ms) if infer_ms else 0.0}
    with open(cache, "wb") as f:
        pickle.dump(series, f)
    return series


def run_counter(series: dict, source: str = "torso", **params) -> JumpCounter:
    counter = JumpCounter(**params)
    w, h = series["width"], series["height"]
    for t, lms in series["frames"]:
        sig = None
        if lms is not None:
            sig = extract_signal([Landmark(*l) for l in lms], w, h, source=source)
        counter.update(t, sig)
    return counter


# --------------------------------------------------------------------- 실행
def evaluate_one(series: dict, gt: list[float] | None, source: str, params: dict,
                 tol: float, show_events: bool) -> dict:
    counter = run_counter(series, source=source, **params)
    pred = [e.t for e in counter.events]
    row = {"count": counter.count}
    if gt is not None:
        tp, fp, fn, offsets = match_events(pred, gt, tol)
        row.update({"gt": len(gt), "tp": tp, "fp": fp, "fn": fn,
                    "offset": statistics.median(offsets) if offsets else 0.0})
    if show_events:
        for e in counter.events:
            print(f"    #{e.index:3d} {e.t:7.2f}s  amp={e.amplitude:.2f} rise={e.rise_time:.2f}s")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="평가할 동영상 파일")
    ap.add_argument("--gt", nargs="*", default=[], help="정답 파일 (영상과 같은 순서, .kva 또는 .html)")
    ap.add_argument("--model", nargs="+", default=["full"], choices=["lite", "full", "heavy"])
    ap.add_argument("--signal", default="torso", choices=["torso", "feet"])
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--min-interval", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.5)
    ap.add_argument("--max-side", type=int, default=0, help="추론 전 긴 변을 이 크기로 축소 (0=원본)")
    ap.add_argument("--sweep", action="store_true", help="threshold 여러 값으로 비교")
    ap.add_argument("--tol", type=float, default=0.35, help="정답 매칭 허용 오차(초)")
    ap.add_argument("--events", action="store_true", help="점프별 시각 출력")
    ap.add_argument("--cache-dir", default=str(ROOT / ".cache"))
    args = ap.parse_args()

    if args.gt and len(args.gt) != len(args.videos):
        raise SystemExit("--gt 파일 수는 영상 수와 같아야 합니다")
    cache_dir = Path(args.cache_dir)
    thresholds = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15] if args.sweep else [args.threshold]

    header = f"{'video':22s} {'model':5s} {'thr':>5s} {'pred':>5s} {'gt':>4s} {'tp':>4s} {'fp':>4s} {'fn':>4s} {'off':>6s}"
    print(header)
    print("-" * len(header))
    totals = {}
    for model in args.model:
        for i, video in enumerate(args.videos):
            series = extract_series(Path(video), model, cache_dir, args.max_side, verbose=False)
            gt = load_ground_truth(args.gt[i]) if args.gt else None
            for thr in thresholds:
                params = dict(threshold=thr, min_interval=args.min_interval, smoothing=args.smoothing)
                row = evaluate_one(series, gt, args.signal, params, args.tol, args.events and not args.sweep)
                if gt is not None:
                    print(f"{Path(video).stem[:22]:22s} {model:5s} {thr:5.2f} {row['count']:5d} {row['gt']:4d} "
                          f"{row['tp']:4d} {row['fp']:4d} {row['fn']:4d} {row['offset']:+6.2f}")
                    key = (model, thr)
                    agg = totals.setdefault(key, {"pred": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0})
                    agg["pred"] += row["count"]; agg["gt"] += row["gt"]
                    agg["tp"] += row["tp"]; agg["fp"] += row["fp"]; agg["fn"] += row["fn"]
                else:
                    print(f"{Path(video).stem[:22]:22s} {model:5s} {thr:5.2f} {row['count']:5d}")
            print(f"  (추론 {series['infer_ms']:.1f} ms/프레임, {series['width']}x{series['height']})")
    if totals:
        print("\n합계 (모델, threshold): 예측/정답  tp fp fn  정밀도 재현율")
        for (model, thr), a in sorted(totals.items()):
            prec = a["tp"] / a["pred"] if a["pred"] else 0.0
            rec = a["tp"] / a["gt"] if a["gt"] else 0.0
            print(f"  {model:5s} {thr:5.2f}: {a['pred']:4d}/{a['gt']:<4d} {a['tp']:4d} {a['fp']:3d} {a['fn']:3d}  "
                  f"{prec:6.1%} {rec:6.1%}")


if __name__ == "__main__":
    main()
