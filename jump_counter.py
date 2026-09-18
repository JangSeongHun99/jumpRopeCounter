"""jump_counter.py - 줄넘기 점프 판정 로직 (MediaPipe/OpenCV 비의존 순수 파이썬).

원리
  1. 포즈 랜드마크에서 몸통 중심(양 어깨·양 엉덩이의 평균)의 세로 좌표를 뽑는다.
  2. 몸통 길이(어깨 중점 ~ 엉덩이 중점)를 기준 길이로 삼아, 카메라와의 거리에
     무관한 상대 단위(몸통 길이 = 1.0)로 움직임을 잰다.
  3. 히스테리시스 피크/밸리 검출: 최저점에서 threshold 이상 올라가면 "상승 중",
     최고점에서 threshold 이상 내려오면 "피크 확정" -> 점프 1회.
     (기준선을 따로 두지 않으므로 사람이 앞뒤로 움직여도 드리프트에 강하다)
  4. 최소 간격(min_interval), 최대 상승 시간(max_rise_time), 최대 진폭(max_amplitude)으로
     걷기·앉았다 일어나기·추적 튐 같은 오검출을 걸러낸다.
  5. 점프인지 확인: 발목이 잘 보이면 최저점->최고점 사이에 한 발이라도 몸통 길이의 feet_lift
     이상 올라갔는지를 기준으로 삼는다 (발이 땅에 붙어 있으면 점프가 아님).
     발이 안 보이면(상반신만 촬영) 머리(코·귀)나 엉덩이가 몸통과 함께 올라갔는지로 대신한다.
     어깨 으쓱, 팔 동작, 발이 땅에 붙은 채 무릎만 굽혔다 펴기는 이 검사에서 걸러진다.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence


class Landmark(NamedTuple):
    """정규화 좌표(0~1)의 가벼운 랜드마크. MediaPipe 객체 대신 스레드 간 전달용."""
    x: float
    y: float
    visibility: float = 1.0


# MediaPipe Pose 33개 랜드마크 중 사용하는 인덱스
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vis(lm) -> float:
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else float(v)


@dataclass
class BodySignal:
    """한 프레임에서 뽑은 신체 신호 (픽셀 단위)."""
    center_y: float                 # 몸통 중심 y (아래로 갈수록 커짐)
    scale: float                    # 몸통 길이 - 정규화 기준
    center_x: float = 0.0
    head_y: Optional[float] = None  # 머리(코·귀 평균) y. 어깨 으쓱과 점프를 구분하는 데 쓴다
    hip_y: Optional[float] = None   # 엉덩이 중점 y (보이지 않으면 None)
    feet_y: Optional[tuple] = None  # (왼발목 y, 오른발목 y). 둘 다 잘 보일 때만. 발이 떴는지 판단


def extract_signal(landmarks: Sequence, width: int, height: int,
                   min_visibility: float = 0.5, source: str = "torso",
                   feet_visibility: float = 0.7) -> Optional[BodySignal]:
    """정규화 랜드마크(x, y, visibility) 목록에서 점프 판정용 신호를 뽑는다.

    source: "torso" (기본, 상반신만 보여도 동작) 또는 "feet" (발목 사용, 전신이 보일 때만).
    feet_visibility: 발목이 이 값 이상으로 확실히 보일 때만 발 기준 검사에 쓴다
    (책상 등에 가려 추정만 된 발목으로 실제 점프를 걸러내지 않도록 기준을 높게 둔다).
    어깨·엉덩이 둘 다 안 보이면 None.
    """
    def pt(i):
        lm = landmarks[i]
        return (lm.x * width, lm.y * height, _vis(lm))

    ls, rs, lh, rh = pt(L_SHOULDER), pt(R_SHOULDER), pt(L_HIP), pt(R_HIP)
    sh_ok = ls[2] >= min_visibility and rs[2] >= min_visibility
    hip_ok = lh[2] >= min_visibility and rh[2] >= min_visibility
    if not sh_ok and not hip_ok:
        return None

    sh_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    if sh_ok and hip_ok:
        scale = _dist(sh_mid, hip_mid)
        center = ((sh_mid[0] + hip_mid[0]) / 2, (sh_mid[1] + hip_mid[1]) / 2)
    elif sh_ok:  # 상반신만 보임: 어깨 너비로 몸통 길이를 추정
        scale = _dist(ls, rs) * 1.25
        center = sh_mid
    else:        # 엉덩이만 보임 (드묾)
        scale = _dist(lh, rh) * 2.5
        center = hip_mid
    if scale < 4:
        return None

    la, ra = pt(L_ANKLE), pt(R_ANKLE)
    feet_ok = la[2] >= feet_visibility and ra[2] >= feet_visibility
    center_y = center[1]
    if source == "feet" and la[2] >= min_visibility and ra[2] >= min_visibility:
        center_y = (la[1] + ra[1]) / 2

    head_pts = [p for p in (pt(NOSE), pt(L_EAR), pt(R_EAR)) if p[2] >= min_visibility]
    head_y = sum(p[1] for p in head_pts) / len(head_pts) if head_pts else None
    return BodySignal(center_y=center_y, scale=scale, center_x=center[0],
                      head_y=head_y, hip_y=hip_mid[1] if hip_ok else None,
                      feet_y=(la[1], ra[1]) if feet_ok else None)


@dataclass
class JumpEvent:
    index: int          # 몇 번째 점프인지 (1부터)
    t: float            # 최고점 시각 (초)
    amplitude: float    # 최저점 대비 상승량 (몸통 길이 단위)
    rise_time: float    # 최저점 -> 최고점 걸린 시간 (초)


class JumpCounter:
    """프레임마다 update(t, signal)를 호출하면 점프를 세어 준다."""

    def __init__(self, threshold: float = 0.05, min_interval: float = 0.20,
                 max_rise_time: float = 0.7, max_amplitude: float = 1.5,
                 smoothing: float = 0.5, lost_timeout: float = 0.5,
                 coherence: float = 0.5, feet_lift: float = 0.03,
                 history_seconds: float = 12.0):
        self.threshold = threshold          # 점프로 인정할 최소 진폭 (몸통 길이 비율)
        self.min_interval = min_interval    # 점프 사이 최소 간격 (초)
        self.max_rise_time = max_rise_time  # 이보다 느리게 올라가면 점프가 아님 (초)
        self.max_amplitude = max_amplitude  # 이보다 크면 추적 오류로 간주
        self.smoothing = smoothing          # EMA 계수 (1이면 평활화 없음)
        self.lost_timeout = lost_timeout    # 이 시간 이상 사람이 안 보이면 상태 초기화
        self.coherence = coherence          # 발이 안 보일 때: 머리/엉덩이 상승량이 몸통 상승량의 이 비율 이상이어야 점프
        self.feet_lift = feet_lift          # 발이 보일 때: 한 발이라도 몸통 길이의 이 비율 이상 올라가야 점프
        self.history = deque()              # (t, 높이) 그래프용
        self._history_seconds = history_seconds
        self.reset()

    # ----------------------------------------------------------------- 상태
    def reset(self):
        self.count = 0
        self.rejected = 0            # 발(또는 머리/엉덩이)이 함께 안 올라가 걸러낸 횟수 (으쓱, 무릎 굽히기 등)
        self.events: list[JumpEvent] = []
        self.last_t: Optional[float] = None
        self.height = 0.0            # 현재 높이 (최근 최저점 대비, 몸통 길이 단위)
        self.tracking = False        # 현재 사람이 보이는지
        self.history.clear()
        self._reset_tracking()

    def _reset_tracking(self):
        self._state = "init"          # init | rising | falling
        self._ext_val = None          # 현재 추적 중인 극값 (상승 중: 최대, 하강 중: 최소)
        self._ext_t = None
        self._valley_val = None       # 마지막으로 확정된 최저점
        self._valley_t = None
        self._last_peak_t = None
        self._smooth = None
        self._scale = None
        self._last_seen_t = None
        self._track = deque()         # (t, 머리 높이, 엉덩이 높이, 왼발 높이, 오른발 높이) 최근 3초

    # ----------------------------------------------------------------- 갱신
    def update(self, t: float, sig: Optional[BodySignal]) -> Optional[JumpEvent]:
        """t: 초 단위 시각(단조 증가). sig: extract_signal 결과 또는 None(사람 없음).
        점프가 확정된 프레임에서 JumpEvent를 돌려준다."""
        self.last_t = t
        if sig is None:
            self.tracking = False
            if self._last_seen_t is not None and t - self._last_seen_t > self.lost_timeout:
                self._reset_tracking()
            return None
        self.tracking = True
        self._last_seen_t = t

        # 기준 길이는 천천히 따라가게 (프레임마다 흔들리지 않도록)
        if self._scale is None:
            self._scale = sig.scale
        else:
            self._scale += 0.05 * (sig.scale - self._scale)
        if self._scale < 1e-3:
            return None

        v = -sig.center_y  # 위쪽이 양수가 되도록 뒤집는다 (픽셀)
        if self._smooth is None:
            self._smooth = v
        else:
            self._smooth += self.smoothing * (v - self._smooth)
        v = self._smooth

        # 머리/엉덩이/발 높이 기록 (피크 확정 시 몸통과 같이 올라갔는지 대조, 위쪽이 양수)
        self._track.append((t, None if sig.head_y is None else -sig.head_y,
                            None if sig.hip_y is None else -sig.hip_y,
                            None if sig.feet_y is None else -sig.feet_y[0],
                            None if sig.feet_y is None else -sig.feet_y[1]))
        while self._track and t - self._track[0][0] > 3.0:
            self._track.popleft()

        # 그래프용 기록 (몸통 길이 단위)
        self.history.append((t, v / self._scale))
        while self.history and t - self.history[0][0] > self._history_seconds:
            self.history.popleft()

        if self._valley_val is not None:
            self.height = (v - self._valley_val) / self._scale
        return self._step(t, v)

    def _step(self, t: float, v: float) -> Optional[JumpEvent]:
        thr = self.threshold * self._scale
        if self._state == "init":
            self._state = "falling"     # 먼저 최저점(서 있는 자세)을 찾는다
            self._ext_val, self._ext_t = v, t
            return None

        if self._state == "rising":
            if v >= self._ext_val:
                self._ext_val, self._ext_t = v, t
            elif self._ext_val - v >= thr:          # 최고점에서 thr 만큼 내려옴 -> 피크 확정
                event = self._on_peak(self._ext_t, self._ext_val)
                self._state = "falling"
                self._ext_val, self._ext_t = v, t
                return event
        else:  # falling
            if v <= self._ext_val:
                self._ext_val, self._ext_t = v, t
            elif v - self._ext_val >= thr:          # 최저점에서 thr 만큼 올라옴 -> 밸리 확정
                self._valley_val, self._valley_t = self._ext_val, self._ext_t
                self._state = "rising"
                self._ext_val, self._ext_t = v, t
        return None

    def _on_peak(self, peak_t: float, peak_val: float) -> Optional[JumpEvent]:
        if self._valley_val is None:
            return None
        amplitude = (peak_val - self._valley_val) / self._scale
        rise_time = peak_t - self._valley_t
        if amplitude > self.max_amplitude:      # 추적이 튄 것
            return None
        if rise_time > self.max_rise_time:      # 천천히 일어난 것 (점프 아님)
            return None
        if self._last_peak_t is not None and peak_t - self._last_peak_t < self.min_interval:
            return None
        if not self._coherent(self._valley_t, peak_t, peak_val - self._valley_val):
            self.rejected += 1                  # 발이 안 뜬 것 (으쓱, 팔 동작, 무릎 굽히기)
            return None
        self._last_peak_t = peak_t
        self.count += 1
        event = JumpEvent(index=self.count, t=peak_t, amplitude=amplitude, rise_time=rise_time)
        self.events.append(event)
        return event

    def _sample_at(self, t: float):
        best = None
        for s in self._track:
            if best is None or abs(s[0] - t) < abs(best[0] - t):
                best = s
        return best

    def _coherent(self, t_valley: float, t_peak: float, body_rise: float) -> bool:
        """최저점->최고점 사이에 몸이 실제로 떴는지.
        발목이 보이면: 한 발이라도 몸통 길이의 feet_lift 이상 올라가야 한다 (발이 붙어 있으면 거부).
        몸통은 무릎 굽힘까지 포함해 오르내리므로 발 상승량을 몸통 상승량과 비율로 비교하지 않는다.
        발이 안 보이면: 머리 또는 엉덩이가 몸통 상승량의 coherence 비율 이상 같이 올라가야 한다.
        아무것도 안 보이면 통과."""
        a, b = self._sample_at(t_valley), self._sample_at(t_peak)
        if a is None or b is None:
            return True
        if None not in (a[3], a[4], b[3], b[4]):
            return max(b[3] - a[3], b[4] - a[4]) >= self.feet_lift * self._scale
        rises = [b[i] - a[i] for i in (1, 2) if a[i] is not None and b[i] is not None]
        if not rises:
            return True
        return max(rises) >= self.coherence * body_rise

    # ----------------------------------------------------------------- 조회
    @property
    def state(self) -> str:
        return self._state

    def cadence(self, now: Optional[float] = None, window: int = 8) -> float:
        """최근 점프들로 계산한 분당 점프 수. 3초 이상 멈추면 0."""
        if len(self.events) < 2:
            return 0.0
        now = self.last_t if now is None else now
        recent = [e.t for e in self.events[-window:]]
        if now is not None and now - recent[-1] > 3.0:
            return 0.0
        span = recent[-1] - recent[0]
        return 60.0 * (len(recent) - 1) / span if span > 0 else 0.0
