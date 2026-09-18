"""tools/test_counter.py - 카운터 로직 단위 테스트 (MediaPipe·카메라 불필요).

합성 랜드마크로 점프, 어깨 으쓱, 무릎 굽히기, 앞뒤로 흔들기, 걷기, 한두 번 튕기기를 만들어
리듬 있는 제자리 점프만 세는지 확인한다.
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
              hips_visible: bool = True, feet_visible: bool = True,
              zoom: float = 0.0, x_off: float = 0.0):
    """정규화 좌표의 33개 랜드마크.
    body_dy: 발을 뺀 몸 전체가 위로 움직인 양, feet_dy: 발이 움직인 양, shoulder_dy: 어깨만 움직인 양,
    zoom: 카메라 쪽으로 다가온 정도(몸이 그만큼 커짐), x_off: 좌우 이동."""
    pts = [Landmark(0.5, 0.5, 0.0)] * 33

    def put(i, x, y, v=0.95, dy=body_dy):
        # zoom: 카메라 쪽으로 다가오면 화면 중심 기준으로 가로세로가 함께 커진다
        pts[i] = Landmark(0.5 + (x - 0.5) * (1 + zoom) + x_off, 0.5 + (y - dy - 0.5) * (1 + zoom), v)

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
          period_s: float = 0.8, bump_s: float = 0.4, drift_x: float = 0.0, zoom_amp: float = 0.0):
    """n번의 반정현파 상승/하강.
    kind: "jump" = 발 포함 전신, "shrug" = 어깨만, "squat" = 발은 땅에 두고 몸만 (무릎 굽혔다 펴기),
          "lean" = 몸이 올라가면서 카메라 쪽으로 다가옴 (앉아서 앞뒤로 흔들기).
    drift_x: 한 주기마다 좌우로 이동하는 양 (걷기)."""
    frames = []
    total = int(n * period_s * FPS) + FPS
    for i in range(total):
        t = i / FPS
        k = int(t // period_s)
        phase = t - k * period_s
        active = k < n and phase < bump_s
        d = amp * math.sin(math.pi * phase / bump_s) if active else 0.0
        frames.append(make_pose(shoulder_dy=d if kind == "shrug" else 0.0,
                                body_dy=d if kind in ("jump", "squat", "lean") else 0.0,
                                feet_dy=d if kind == "jump" else 0.0,
                                hips_visible=hips_visible, feet_visible=feet_visible,
                                zoom=(d / amp * zoom_amp) if (active and amp) else 0.0,
                                x_off=drift_x * min(t / period_s, n)))
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
        ("빠른 점프 12회 (분당 200회)", bumps(12, 0.03, "jump", period_s=0.3, bump_s=0.24), 12),
        ("점프 3회 (리듬 확인 최소 횟수)", bumps(3, 0.04, "jump"), 3),
        ("점프 2회만 (리듬 미확인)", bumps(2, 0.04, "jump"), 0),
        ("한 번 튕기기", bumps(1, 0.06, "jump"), 0),
        ("어깨 으쓱 5회 (전신 보임)", bumps(5, 0.04, "shrug"), 0),
        ("어깨 으쓱 5회 (상반신만 보임)", bumps(5, 0.04, "shrug", hips_visible=False, feet_visible=False), 0),
        ("발 붙이고 무릎만 굽혔다 펴기 5회", bumps(5, 0.04, "squat"), 0),
        ("앉아서 앞뒤로 흔들기 5회 (상반신, 몸 크기 25% 변화)",
         bumps(5, 0.04, "lean", hips_visible=False, feet_visible=False, zoom_amp=0.25), 0),
        ("걷기 5걸음 (몸이 들썩이며 좌우 이동)", bumps(5, 0.04, "jump", drift_x=0.3), 0),
        ("가만히 있기", bumps(0, 0.0, "jump"), 0),
    ]
    ok = True
    for name, seq, expect in cases:
        counter = run(seq)
        passed = counter.count == expect
        ok = ok and passed
        rej = counter.rejected
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {counter.count}회 (기대 {expect}; 걸러냄 "
              f"발 {rej['feet']}, 이동 {rej['motion']}, 리듬 {rej['rhythm']}, 대기 {len(counter.pending)})")
    print("모두 통과" if ok else "실패 있음")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
