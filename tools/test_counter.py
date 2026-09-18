"""tools/test_counter.py - 카운터 로직 단위 테스트 (MediaPipe·카메라 불필요).

합성 랜드마크로 점프, 어깨 으쓱, 발을 붙인 채 무릎 굽히기를 만들어 점프만 세는지 확인한다.
  python tools/test_counter.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jump_counter import JumpCounter, Landmark, extract_signal  # noqa: E402

W, H, FPS = 1280, 720, 30


def make_pose(shoulder_dy: float = 0.0, body_dy: float = 0.0, feet_dy: float = 0.0,
              hips_visible: bool = True, feet_visible: bool = True):
    """정규화 좌표의 33개 랜드마크.
    body_dy: 발을 뺀 몸 전체가 위로 움직인 양, feet_dy: 발이 움직인 양, shoulder_dy: 어깨만 움직인 양."""
    pts = [Landmark(0.5, 0.5, 0.0)] * 33

    def put(i, x, y, v=0.95, dy=body_dy):
        pts[i] = Landmark(x, y - dy, v)

    put(0, 0.50, 0.25)                        # 코
    put(7, 0.47, 0.26)                        # 왼쪽 귀
    put(8, 0.53, 0.26)                        # 오른쪽 귀
    put(11, 0.42, 0.40 - shoulder_dy)         # 어깨
    put(12, 0.58, 0.40 - shoulder_dy)
    hv = 0.95 if hips_visible else 0.1
    put(23, 0.45, 0.62, hv)                   # 엉덩이
    put(24, 0.55, 0.62, hv)
    fv = 0.9 if feet_visible else 0.1
    put(27, 0.45, 0.95, fv, dy=feet_dy)       # 발목
    put(28, 0.55, 0.95, fv, dy=feet_dy)
    return pts


def bumps(n: int, amp: float, kind: str, hips_visible: bool = True, feet_visible: bool = True,
          period_s: float = 0.8, bump_s: float = 0.4):
    """n번의 반정현파 상승/하강.
    kind: "jump" = 발 포함 전신, "shrug" = 어깨만, "squat" = 발은 땅에 두고 몸만 (무릎 굽혔다 펴기)."""
    frames = []
    total = int(n * period_s * FPS) + FPS
    for i in range(total):
        t = i / FPS
        k = int(t // period_s)
        phase = t - k * period_s
        d = amp * math.sin(math.pi * phase / bump_s) if (k < n and phase < bump_s) else 0.0
        frames.append(make_pose(shoulder_dy=d if kind == "shrug" else 0.0,
                                body_dy=d if kind in ("jump", "squat") else 0.0,
                                feet_dy=d if kind == "jump" else 0.0,
                                hips_visible=hips_visible, feet_visible=feet_visible))
    return frames


def run(seq):
    counter = JumpCounter()
    for i, lms in enumerate(seq):
        counter.update(i / FPS, extract_signal(lms, W, H))
    return counter


def main() -> int:
    cases = [
        ("점프 5회 (전신 보임)", bumps(5, 0.04, "jump"), 5),
        ("점프 5회 (발 안 보임)", bumps(5, 0.04, "jump", feet_visible=False), 5),
        ("점프 5회 (상반신만 보임)", bumps(5, 0.04, "jump", hips_visible=False, feet_visible=False), 5),
        ("작은 점프 5회 (진폭 몸통의 7%)", bumps(5, 0.015, "jump"), 5),
        ("어깨 으쓱 5회 (전신 보임)", bumps(5, 0.04, "shrug"), 0),
        ("어깨 으쓱 5회 (상반신만 보임)", bumps(5, 0.04, "shrug", hips_visible=False, feet_visible=False), 0),
        ("발 붙이고 무릎만 굽혔다 펴기 5회 (전신 보임)", bumps(5, 0.04, "squat"), 0),
        ("가만히 있기", bumps(0, 0.0, "jump"), 0),
    ]
    ok = True
    for name, seq, expect in cases:
        counter = run(seq)
        passed = counter.count == expect
        ok = ok and passed
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {counter.count}회 (기대 {expect}, 걸러냄 {counter.rejected})")
    print("모두 통과" if ok else "실패 있음")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
