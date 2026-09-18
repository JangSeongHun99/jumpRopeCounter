#!/usr/bin/env python3
"""jump_rope_counter.py - 노트북 카메라로 줄넘기 횟수를 세는 프로그램.

사용법
  python jump_rope_counter.py                   # 기본 카메라(0번)로 시작
  python jump_rope_counter.py --camera 1        # 다른 카메라
  python jump_rope_counter.py --video 파일.mp4   # 녹화된 영상 분석
  python jump_rope_counter.py --list-cameras    # 연결된 카메라 찾기

화면 단축키
  q / ESC  종료          r  카운트 초기화       space  일시정지
  s  세션 저장           + / -  민감도 올리기/내리기
  m  좌우반전 토글       d  스켈레톤·그래프 표시 토글

구조 (성능)
  카메라 읽기 스레드  -> 항상 최신 프레임만 유지 (지연 최소화)
  포즈 추론 스레드    -> MediaPipe 추론 + 점프 판정 (추론 중 GIL 해제되어 병렬 동작)
  메인 스레드         -> 화면 그리기와 키 입력만 담당. 추론이 느려도 화면은 끊기지 않는다.
판정 방식은 jump_counter.py 상단 설명 참고.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from jump_counter import JumpCounter, extract_signal
from pose_backend import MODEL_NAMES, create_landmarker, detect_landmarks, draw_skeleton, landmarks_to_tuples

ROOT = Path(__file__).resolve().parent
WINDOW = "Jump Rope Counter"
KEY_HELP = "q quit   r reset   space pause   s save   +/- sensitivity   m mirror   d overlay"


# ===================================================================== 입력
def open_camera(index: int, width: int, height: int):
    """Windows에서는 DirectShow -> MSMF -> 자동 순서로 시도한다. (cap, 백엔드 이름) 또는 (None, None)."""
    if sys.platform == "win32":
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "auto")]
    else:
        backends = [(cv2.CAP_ANY, "auto")]
    for api, name in backends:
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            continue
        # 내장 웹캠은 MJPG 로 받아야 720p에서도 30fps가 나오는 경우가 많다 (지원 안 하면 무시됨)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):  # 첫 몇 프레임은 실패할 수 있다
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, name
        cap.release()
    return None, None


def list_cameras(max_index: int = 6) -> None:
    print("카메라를 찾는 중... (몇 초 걸릴 수 있습니다)")
    found = []
    for i in range(max_index):
        cap, backend = open_camera(i, 0, 0)
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  --camera {i} : {w}x{h} ({backend})")
            found.append(i)
            cap.release()
    if not found:
        print("  연결된 카메라를 찾지 못했습니다. 카메라 개인정보 설정(데스크톱 앱 허용)과 다른 프로그램의 사용 여부를 확인하세요.")


class CameraReader(threading.Thread):
    """카메라에서 계속 프레임을 읽어 최신 프레임만 유지한다 (오래된 프레임은 버림)."""

    def __init__(self, cap):
        super().__init__(daemon=True, name="camera-reader")
        self.cap = cap
        self._cv = threading.Condition()
        self._frame = None
        self.seq = 0
        self.running = True
        self.failures = 0

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.failures += 1
                if self.failures > 100:      # 약 1초 이상 계속 실패하면 포기
                    self.running = False
                    with self._cv:
                        self._cv.notify_all()
                    break
                time.sleep(0.01)
                continue
            self.failures = 0
            with self._cv:
                self._frame = frame
                self.seq += 1
                self._cv.notify_all()

    def wait_new(self, last_seq: int, timeout: float):
        """last_seq 이후의 새 프레임을 timeout 초까지 기다린다. (frame, seq) 또는 (None, last_seq).
        폴링 대신 대기하므로 메인 스레드가 추론 스레드와 CPU/GIL을 다투지 않는다."""
        with self._cv:
            if self.seq == last_seq and self.running:
                self._cv.wait(timeout)
            if self.seq != last_seq:
                return self._frame, self.seq
            return None, last_seq

    def stop(self):
        self.running = False
        with self._cv:
            self._cv.notify_all()


# ===================================================================== 추론
def downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame
    s = max_side / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


class PoseWorker(threading.Thread):
    """포즈 추론 + 점프 판정. 카메라 모드에서는 스레드로 돌고, 영상 모드에서는 process()를 직접 부른다."""

    def __init__(self, landmarker, counter: JumpCounter, lock: threading.Lock,
                 source: str, infer_size: int):
        super().__init__(daemon=True, name="pose-worker")
        self.landmarker = landmarker
        self.counter = counter
        self.lock = lock
        self.source = source
        self.infer_size = infer_size
        self._cv = threading.Condition()
        self._pending = None
        self._running = True
        self._last_ts = -1
        self.latest_landmarks = None     # 정규화 좌표 튜플 목록 (그리기용)
        self.events = deque()            # 메인 스레드가 꺼내 가는 새 점프 이벤트
        self.infer_ms = 0.0
        self.processed = 0
        self.dropped = 0

    def submit(self, frame, t: float):
        with self._cv:
            if self._pending is not None:
                self.dropped += 1        # 아직 처리 못 한 프레임은 버리고 최신 것으로 교체
            self._pending = (frame, t)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify()

    def run(self):
        while True:
            with self._cv:
                while self._pending is None and self._running:
                    self._cv.wait(0.1)
                if not self._running:
                    return
                frame, t = self._pending
                self._pending = None
            try:
                self.process(frame, t)
            except Exception as exc:  # 추론 오류로 프로그램 전체가 죽지 않게
                print(f"[추론 오류] {exc}")

    def process(self, frame, t: float):
        small = downscale(frame, self.infer_size)
        ts = int(t * 1000)
        if ts <= self._last_ts:          # MediaPipe VIDEO 모드는 단조 증가 타임스탬프 필요
            ts = self._last_ts + 1
        self._last_ts = ts
        t0 = time.perf_counter()
        lms = detect_landmarks(self.landmarker, small, ts)
        dt = (time.perf_counter() - t0) * 1000
        self.infer_ms = dt if self.processed == 0 else self.infer_ms * 0.9 + dt * 0.1
        self.processed += 1
        tuples = landmarks_to_tuples(lms) if lms else None
        sig = None
        if tuples:
            sig = extract_signal(tuples, small.shape[1], small.shape[0], source=self.source)
        with self.lock:
            event = self.counter.update(t, sig)
        self.latest_landmarks = tuples
        if event is not None:
            self.events.append(event)
        return event


# ===================================================================== 그리기
def put_text(img, text, org, scale, color=(255, 255, 255), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(img, x1, y1, x2, y2, alpha=0.55):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0)


def draw_signal_graph(img, history, events, threshold, t_now, window_s=6.0):
    """최근 6초의 몸통 높이(몸통 길이 단위) 그래프와 점프 마커."""
    h, w = img.shape[:2]
    s = h / 720.0
    gh = int(100 * s)
    x1, y1, x2, y2 = 12, h - 12 - gh, w - 12, h - 12
    draw_panel(img, x1, y1, x2, y2, 0.5)
    put_text(img, "body height (last 6 s)", (x1 + 8, y1 + int(16 * s)), 0.45 * s, (200, 200, 200), 1)
    t_start = t_now - window_s
    pts = [(t, v) for t, v in history if t >= t_start]
    if len(pts) < 2:
        return
    vals = [v for _, v in pts]
    vmin, vmax = min(vals), max(vals)
    rng = max(vmax - vmin, threshold * 3)
    lo = (vmax + vmin) / 2 - rng / 2

    def X(t):
        return int(x1 + 8 + (t - t_start) / window_s * (x2 - x1 - 16))

    def Y(v):
        return int(y2 - 8 - (v - lo) / rng * (gh - 24))

    poly = np.array([(X(t), Y(v)) for t, v in pts], dtype=np.int32)
    cv2.polylines(img, [poly], False, (0, 220, 255), max(1, int(2 * s)), cv2.LINE_AA)
    for ev in events:
        if ev.t >= t_start:
            cv2.line(img, (X(ev.t), y1 + int(20 * s)), (X(ev.t), y2 - 4), (80, 255, 120), 1, cv2.LINE_AA)
    # threshold 눈금 (이만큼 오르내려야 점프로 인정)
    xb = x2 - int(14 * s)
    yb = Y(lo + rng * 0.15)
    cv2.line(img, (xb, yb), (xb, Y(lo + rng * 0.15 + threshold)), (255, 255, 255), 2)
    put_text(img, "thr", (xb - int(30 * s), yb), 0.4 * s, (255, 255, 255), 1)


# ===================================================================== 앱
class CounterApp:
    def __init__(self, args, cap, is_camera: bool, src_fps: float, counter: JumpCounter,
                 lock: threading.Lock, worker: PoseWorker):
        self.args = args
        self.cap = cap
        self.is_camera = is_camera
        self.src_fps = src_fps
        self.counter = counter
        self.lock = lock
        self.worker = worker
        self.mirror = is_camera and not args.no_mirror
        self.display = not args.no_display
        self.overlay = True
        self.paused = False
        self.pause_started = None
        self.paused_total = 0.0
        self.t0 = time.monotonic()
        self.t_last = 0.0
        self.session_t0 = 0.0
        self.session_started = datetime.now()
        self.flash_t = -10.0
        self.fps = 0.0
        self._fps_last = None
        self.writer = None
        self.last_frame = None
        self.saved_paths = []
        self.slow_hint_shown = False

    # ------------------------------------------------------------ 시간
    def elapsed(self, t: float) -> float:
        e = t - self.session_t0 - self.paused_total
        if self.paused and self.pause_started is not None:
            e -= t - self.pause_started
        return max(0.0, e)

    def _tick_fps(self):
        now = time.perf_counter()
        if self._fps_last is not None:
            dt = now - self._fps_last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else self.fps * 0.9 + inst * 0.1
        self._fps_last = now

    # ------------------------------------------------------------ 루프
    def run(self):
        if self.display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if self.is_camera:
            self._run_camera()
        else:
            self._run_video()

    def _run_camera(self):
        reader = CameraReader(self.cap)
        reader.start()
        self.worker.start()
        last_seq = 0
        try:
            while True:
                frame, seq = reader.wait_new(last_seq, timeout=0.02)
                if frame is not None:
                    last_seq = seq
                    if self.mirror:
                        frame = cv2.flip(frame, 1)
                    t = time.monotonic() - self.t0
                    self.t_last = t
                    if not self.paused:
                        self.worker.submit(frame, t)
                    self.last_frame = frame
                    self._tick_fps()
                    self._render(frame, t)
                elif not reader.running:
                    print("[오류] 카메라 연결이 끊겼습니다 (프레임을 받지 못함).")
                    break
                if not self._handle_keys():
                    break
        finally:
            reader.stop()
            self.worker.stop()

    def _run_video(self):
        idx = 0
        t = 0.0
        while True:
            if not self.paused:
                ok, frame = self.cap.read()
                if not ok:
                    break
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                t = idx / self.src_fps
                idx += 1
                self.t_last = t
                self.worker.process(frame, t)          # 동기 처리: 프레임 누락 없음
                self.last_frame = frame
                self._tick_fps()
                if self.display or self.args.record:
                    self._render(frame, t)
            if self.display:
                if not self._handle_keys(wait_ms=30 if self.paused else 1):
                    break
            elif self.paused:
                self.paused = False

    # ------------------------------------------------------------ 이벤트
    def _drain_events(self, t: float):
        while self.worker.events:
            ev = self.worker.events.popleft()
            self.flash_t = t
            if self.args.verbose:
                print(f"  #{ev.index:4d}  {ev.t - self.session_t0:7.2f}s  amp={ev.amplitude:.2f}")
        if self.is_camera and not self.slow_hint_shown and self.worker.processed >= 90:
            self.slow_hint_shown = True
            if self.worker.infer_ms > 45:
                print(f"[힌트] 포즈 추론이 느립니다({self.worker.infer_ms:.0f} ms/프레임). "
                      f"--model lite 또는 --infer-size 640, --width 640 --height 480 을 써 보세요.")

    def _handle_keys(self, wait_ms: int = 1) -> bool:
        if not self.display:
            return True
        key = cv2.waitKey(wait_ms) & 0xFF
        t = self.t_last
        if key in (ord("q"), 27):
            return False
        if key == ord("r"):
            with self.lock:
                self.counter.reset()
            self.worker.events.clear()
            self.session_t0 = t
            self.session_started = datetime.now()
            self.paused_total = 0.0
            self.pause_started = t if self.paused else None
            print("[초기화] 카운트를 0으로 되돌렸습니다.")
        elif key == ord(" "):
            self.paused = not self.paused
            if self.paused:
                self.pause_started = t
            elif self.pause_started is not None:
                self.paused_total += t - self.pause_started
                self.pause_started = None
        elif key == ord("s"):
            self.save_session(force=True)
        elif key in (ord("+"), ord("=")):
            self.counter.threshold = round(max(0.01, self.counter.threshold - 0.01), 3)
            print(f"[민감도] threshold {self.counter.threshold:.2f} (작을수록 민감)")
        elif key == ord("-"):
            self.counter.threshold = round(min(0.5, self.counter.threshold + 0.01), 3)
            print(f"[민감도] threshold {self.counter.threshold:.2f} (작을수록 민감)")
        elif key == ord("m"):
            self.mirror = not self.mirror
        elif key == ord("d"):
            self.overlay = not self.overlay
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            pass
        return True

    # ------------------------------------------------------------ 화면
    def _render(self, frame, t: float):
        self._drain_events(t)
        img = frame.copy()
        h, w = img.shape[:2]
        s = h / 720.0
        with self.lock:
            count = self.counter.count
            rpm = self.counter.cadence(t)
            tracking = self.counter.tracking
            history = list(self.counter.history)
            events = self.counter.events[-80:]
            threshold = self.counter.threshold
        lms = self.worker.latest_landmarks
        if self.overlay and lms:
            draw_skeleton(img, lms)

        # 왼쪽 위: 점프 수 / 분당 횟수 / 시간
        flash = (t - self.flash_t) < 0.3
        pw, ph = int(300 * s), int(190 * s)
        draw_panel(img, 12, 12, 12 + pw, 12 + ph)
        put_text(img, "JUMPS", (26, 12 + int(32 * s)), 0.75 * s, (190, 190, 190), max(1, int(2 * s)))
        put_text(img, str(count), (22, 12 + int(135 * s)), (3.7 if flash else 3.3) * s,
                 (80, 255, 120) if flash else (255, 255, 255), max(2, int(7 * s)), cv2.FONT_HERSHEY_DUPLEX)
        mm, ss = divmod(int(self.elapsed(t)), 60)
        put_text(img, f"{rpm:3.0f} /min    {mm:02d}:{ss:02d}", (26, 12 + int(175 * s)), 0.8 * s,
                 (220, 220, 220), max(1, int(2 * s)))
        status = (f"pose {self.worker.infer_ms:.0f} ms | {self.fps:.0f} fps | {self.args.model} | "
                  f"thr {threshold:.2f}" + (" | mirror" if self.mirror else ""))
        put_text(img, status, (14, 12 + ph + int(24 * s)), 0.55 * s, (200, 200, 200), 1)

        if self.overlay:
            draw_signal_graph(img, history, events, threshold, t)
        help_y = h - 12 - int(100 * s) - int(8 * s) if self.overlay else h - int(10 * s)
        put_text(img, KEY_HELP, (14, help_y), 0.48 * s, (200, 200, 200), 1)

        msg = None
        if self.paused:
            msg = "PAUSED"
        elif not tracking:
            msg = "No person detected - stand back so your upper body is visible"
        if msg:
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, 2)[0]
            put_text(img, msg, ((w - size[0]) // 2, h // 2), 0.9 * s, (0, 200, 255), 2)

        if self.args.record:
            if self.writer is None:
                fps = self.src_fps if not self.is_camera else 30.0
                self.writer = cv2.VideoWriter(self.args.record, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
            self.writer.write(img)
        if self.display:
            cv2.imshow(WINDOW, img)

    # ------------------------------------------------------------ 저장
    def save_session(self, force: bool = False):
        with self.lock:
            events = [dict(index=e.index, time_s=round(e.t - self.session_t0, 3),
                           amplitude=round(e.amplitude, 3), rise_time_s=round(e.rise_time, 3))
                      for e in self.counter.events]
            count = self.counter.count
        if count == 0 and not force:
            return None
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.session_started.strftime("%Y%m%d_%H%M%S")
        duration = self.elapsed(self.t_last)
        csv_path = save_dir / f"session_{stamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "time_s", "amplitude", "rise_time_s"])
            writer.writeheader()
            writer.writerows(events)
        summary = {
            "started_at": self.session_started.isoformat(timespec="seconds"),
            "source": self.args.video or f"camera {self.args.camera}",
            "total_jumps": count,
            "duration_s": round(duration, 1),
            "avg_per_min": round(count / duration * 60, 1) if duration > 0 else 0.0,
            "params": {"model": self.args.model, "threshold": self.counter.threshold,
                       "min_interval": self.counter.min_interval, "signal": self.args.signal},
        }
        json_path = save_dir / f"session_{stamp}.json"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.saved_paths = [csv_path, json_path]
        print(f"[저장] {csv_path}  /  {json_path.name}")
        return summary

    def summary_text(self) -> str:
        with self.lock:
            count = self.counter.count
        duration = self.elapsed(self.t_last)
        mm, ss = divmod(int(duration), 60)
        per_min = count / duration * 60 if duration > 0 else 0.0
        lines = [f"점프 {count}회 / {mm:02d}:{ss:02d} / 평균 {per_min:.0f}회/분"]
        if self.worker.processed:
            lines.append(f"추론 {self.worker.processed}프레임, 평균 {self.worker.infer_ms:.1f} ms"
                         + (f", 건너뜀 {self.worker.dropped}" if self.worker.dropped else ""))
        return "\n".join(lines)


# ===================================================================== 시작
def parse_args():
    p = argparse.ArgumentParser(description="노트북 카메라 줄넘기 카운터",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("화면 단축키")[1] if "화면 단축키" in __doc__ else "")
    p.add_argument("--camera", "-c", type=int, default=0, help="카메라 번호 (기본 0 = 노트북 내장 카메라)")
    p.add_argument("--video", "-v", help="카메라 대신 동영상 파일 사용")
    p.add_argument("--list-cameras", action="store_true", help="연결된 카메라 번호를 찾아 출력하고 종료")
    p.add_argument("--model", choices=MODEL_NAMES, default="full",
                   help="포즈 모델: lite(가장 빠름) / full(기본) / heavy(가장 정확, 느림)")
    p.add_argument("--width", type=int, default=1280, help="카메라 요청 가로 해상도 (기본 1280)")
    p.add_argument("--height", type=int, default=720, help="카메라 요청 세로 해상도 (기본 720)")
    p.add_argument("--infer-size", type=int, default=0,
                   help="추론 전 긴 변을 이 크기로 축소 (0=원본). 느린 노트북이면 640 권장")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="점프로 인정할 최소 진폭, 몸통 길이 대비 비율 (작을수록 민감, 기본 0.05)")
    p.add_argument("--min-interval", type=float, default=0.2, help="점프 사이 최소 간격(초), 기본 0.2")
    p.add_argument("--smoothing", type=float, default=0.5, help="신호 평활화 계수 0~1 (1=없음), 기본 0.5")
    p.add_argument("--signal", choices=["torso", "feet"], default="torso",
                   help="움직임을 잴 부위: torso(몸통, 기본) / feet(발목, 전신이 보일 때)")
    p.add_argument("--no-mirror", action="store_true", help="카메라 화면 좌우반전 끄기")
    p.add_argument("--no-display", action="store_true", help="창 없이 실행 (동영상 일괄 처리용)")
    p.add_argument("--record", help="주석이 그려진 결과 영상을 이 경로(.mp4)에 저장")
    p.add_argument("--save-dir", default=str(ROOT / "sessions"), help="세션 기록(CSV/JSON) 저장 폴더")
    p.add_argument("--no-save", action="store_true", help="종료 시 세션 기록을 저장하지 않음")
    p.add_argument("--verbose", action="store_true", help="점프마다 콘솔에 한 줄씩 출력")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if args.list_cameras:
        list_cameras()
        return

    is_camera = args.video is None
    if is_camera:
        cap, backend = open_camera(args.camera, args.width, args.height)
        if cap is None:
            print(f"[오류] {args.camera}번 카메라를 열 수 없습니다.\n"
                  f"  - python jump_rope_counter.py --list-cameras 로 번호를 확인하세요\n"
                  f"  - 다른 프로그램(Zoom, 카메라 앱 등)이 카메라를 쓰고 있으면 닫으세요\n"
                  f"  - Windows 설정 > 개인 정보 > 카메라 > '데스크톱 앱이 카메라에 액세스하도록 허용'을 켜세요")
            sys.exit(1)
        src_desc = f"카메라 {args.camera} ({backend})"
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[오류] 영상을 열 수 없습니다: {args.video}")
            sys.exit(1)
        src_desc = args.video
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[입력] {src_desc}  {w}x{h} @ {src_fps:.0f}fps")

    landmarker = create_landmarker(args.model)
    print(f"[모델] pose_landmarker_{args.model}  |  threshold {args.threshold}  |  signal {args.signal}")
    if is_camera:
        print("카메라 앞에서 상반신(어깨~엉덩이)이 보이도록 서세요. q 로 종료합니다.")

    counter = JumpCounter(threshold=args.threshold, min_interval=args.min_interval, smoothing=args.smoothing)
    lock = threading.Lock()
    worker = PoseWorker(landmarker, counter, lock, args.signal, args.infer_size)
    # 추론 스레드가 GIL을 빨리 돌려받도록 스레드 전환 간격을 줄인다 (기본 5 ms -> 1 ms).
    # 30fps 웹캠 모사 테스트에서 프레임당 추론 25 ms -> 17 ms 로 줄었다.
    sys.setswitchinterval(0.001)
    app = CounterApp(args, cap, is_camera, src_fps, counter, lock, worker)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        cap.release()
        if app.writer is not None:
            app.writer.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("\n=== 결과 ===")
        print(app.summary_text())
        if not args.no_save:
            app.save_session()


if __name__ == "__main__":
    main()
