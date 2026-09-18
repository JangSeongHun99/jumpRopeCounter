"""jump_counter.py - 줄넘기 점프 판정 로직 (MediaPipe/OpenCV 비의존 순수 파이썬).

원리
  1. 포즈 랜드마크에서 어깨 중점의 세로 좌표를 뽑는다. (엉덩이가 화면 아래 가장자리에서
     보였다 안 보였다 해도 신호가 튀지 않도록 어깨만 쓴다. 화면 밖으로 추정된 관절은 무시한다.)
  2. 몸통 길이(어깨 중점 ~ 엉덩이 중점, 엉덩이가 안 보이면 어깨 너비 x 1.25)를 기준 길이로 삼아,
     카메라와의 거리에 무관한 상대 단위(몸통 길이 = 1.0)로 움직임을 잰다.
  3. 히스테리시스 피크/밸리 검출: 최저점에서 threshold 이상 올라가면 "상승 중",
     최고점에서 threshold 이상 내려오면 "피크 확정" -> 점프 후보 1회.
     (기준선을 따로 두지 않으므로 사람이 앞뒤로 움직여도 드리프트에 강하다)
  4. 최소 간격(min_interval), 최대 상승 시간(max_rise_time), 최대 진폭(max_amplitude)으로
     앉았다 일어나기·추적 튐 같은 오검출을 걸러낸다.
  5. 발목이 화면 안에 잘 보이면 최저점->최고점 사이에 한 발이라도 몸통 길이의 feet_lift 이상
     올라가야 점프다 (발이 땅에 붙어 있으면 아님). 발이 안 보이면(상반신만 촬영) 머리(코·귀)나
     엉덩이가 어깨와 함께 올라갔는지로 대신한다. 어깨 으쓱, 무릎만 굽혔다 펴기가 걸러진다.
  6. 제자리 점프는 좌우로 움직이지 않고 카메라와의 거리(= 몸 크기)도 변하지 않는다.
     최저점->최고점 사이 좌우 이동이 몸통 길이의 max_shift 이상이거나(걷기), 몸통 길이가
     max_scale_change 이상(엉덩이가 안 보이면 어깨 너비가 max_width_change 이상) 변하면
     (카메라 쪽으로 기울이기) 점프가 아니다. 크기는 3프레임 중앙값을 써서 관절 튐을 줄인다.
  7. 줄넘기는 리듬이 있다. 후보가 비슷한 간격(period_range 안, 편차 period_tolerance 이하)으로
     rhythm_min 회 이어져야 세기 시작하고, 그때 앞의 후보들도 한꺼번에 반영한다.
     리듬이 잡힌 뒤에는 최근 주기와 비슷한 간격(0.65~1.5배)으로 오는 후보만 바로 세고,
     간격이 크게 어긋나면(걸린 뒤 허둥대는 걸음 등) 다시 리듬 확인으로 돌아간다.
     자세 바꾸기, 몸 흔들기, 한두 번 튕기기 같은 산발적 움직임은 여기서 걸러진다.
  8. 줄에 걸리면 리듬이 끊긴다. 줄은 30fps에서 보이지 않으므로, rhythm_break 초 넘게 쉬었다가
     다시 리듬이 잡히면 그 직전 점프(걸린 시도)를 1회 빼고 misses 를 1 올린다 (miss_correction).
     스스로 쉬었다 재개해도 1회가 빠지는 대가가 있다. 마지막에 그냥 멈추면 빠지지 않는다.
     streak(현재 연속)와 best_streak(최고 연속)도 같이 센다.
  lenient=True 로 만들면 5~8번 검사를 끄고 몸이 오르내린 횟수만 센다.
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

# 사람이 볼 수 있는 거부 이유 (화면 표시용)
REASONS = {
    "feet": "feet did not lift",
    "head": "head/hips did not rise with shoulders",
    "shift": "moved sideways",
    "scale": "moved toward/away from camera",
    "slow": "rose too slowly",
    "soon": "too soon after last jump",
    "big": "tracking jumped",
}


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vis(lm) -> float:
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else float(v)


@dataclass
class BodySignal:
    """한 프레임에서 뽑은 신체 신호 (픽셀 단위)."""
    center_y: float                 # 어깨 중점 y (아래로 갈수록 커짐)
    scale: float                    # 몸통 길이 - 정규화 기준
    center_x: float = 0.0           # 어깨 중점 x (좌우 이동 검사용)
    width: Optional[float] = None   # 어깨 너비 (카메라와의 거리 변화 검사용, 엉덩이가 안 보일 때)
    torso: Optional[float] = None   # 몸통 길이 (카메라와의 거리 변화 검사용, 더 안정적)
    head_y: Optional[float] = None  # 머리(코·귀 평균) y. 어깨 으쓱과 점프를 구분하는 데 쓴다
    hip_y: Optional[float] = None   # 엉덩이 중점 y (보이지 않으면 None)
    feet_y: Optional[tuple] = None  # (왼발목 y, 오른발목 y). 둘 다 화면 안에 잘 보일 때만


def extract_signal(landmarks: Sequence, width: int, height: int,
                   min_visibility: float = 0.5, source: str = "torso",
                   feet_visibility: float = 0.7) -> Optional[BodySignal]:
    """정규화 랜드마크(x, y, visibility) 목록에서 점프 판정용 신호를 뽑는다.

    source: "torso" (기본, 상반신만 보여도 동작) 또는 "feet" (발목 사용, 전신이 보일 때만).
    feet_visibility: 발목이 이 값 이상으로 확실히 보일 때만 발 기준 검사에 쓴다.
    화면 밖(정규화 좌표 0~1 밖)으로 추정된 관절은 가시성과 무관하게 보이지 않는 것으로 친다.
    어깨·엉덩이 둘 다 안 보이면 None.
    """
    def pt(i):
        lm = landmarks[i]
        v = _vis(lm)
        if not (-0.02 <= lm.x <= 1.02 and -0.02 <= lm.y <= 1.02):
            v = 0.0
        return (lm.x * width, lm.y * height, v)

    ls, rs, lh, rh = pt(L_SHOULDER), pt(R_SHOULDER), pt(L_HIP), pt(R_HIP)
    sh_ok = ls[2] >= min_visibility and rs[2] >= min_visibility
    hip_ok = lh[2] >= min_visibility and rh[2] >= min_visibility
    if not sh_ok and not hip_ok:
        return None

    sh_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    shoulder_w = _dist(ls, rs) if sh_ok else None
    if sh_ok and hip_ok:
        scale = _dist(sh_mid, hip_mid)
    elif sh_ok:  # 상반신만 보임: 어깨 너비로 몸통 길이를 추정
        scale = shoulder_w * 1.25
    else:        # 엉덩이만 보임 (드묾)
        scale = _dist(lh, rh) * 2.5
    if scale < 4:
        return None
    center = sh_mid if sh_ok else hip_mid

    la, ra = pt(L_ANKLE), pt(R_ANKLE)
    feet_ok = la[2] >= feet_visibility and ra[2] >= feet_visibility
    center_y = center[1]
    if source == "feet" and la[2] >= min_visibility and ra[2] >= min_visibility:
        center_y = (la[1] + ra[1]) / 2

    head_pts = [p for p in (pt(NOSE), pt(L_EAR), pt(R_EAR)) if p[2] >= min_visibility]
    head_y = sum(p[1] for p in head_pts) / len(head_pts) if head_pts else None
    return BodySignal(center_y=center_y, scale=scale, center_x=center[0], width=shoulder_w,
                      torso=scale if (sh_ok and hip_ok) else None,
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
                 max_rise_time: float = 0.5, max_amplitude: float = 1.5,
                 smoothing: float = 0.5, lost_timeout: float = 0.5,
                 coherence: float = 0.5, feet_lift: float = 0.03,
                 max_shift: float = 0.5, max_scale_change: float = 0.2, max_width_change: float = 0.4,
                 rhythm_min: int = 3, period_range: tuple = (0.25, 1.5),
                 period_tolerance: float = 0.5, rhythm_break: float = 2.0,
                 miss_correction: bool = True, lenient: bool = False, debug: bool = False,
                 history_seconds: float = 12.0):
        self.threshold = threshold          # 점프로 인정할 최소 진폭 (몸통 길이 비율)
        self.min_interval = min_interval    # 점프 사이 최소 간격 (초)
        self.max_rise_time = max_rise_time  # 이보다 느리게 올라가면 점프가 아님 (초)
        self.max_amplitude = max_amplitude  # 이보다 크면 추적 오류로 간주
        self.smoothing = smoothing          # EMA 계수 (1이면 평활화 없음)
        self.lost_timeout = lost_timeout    # 이 시간 이상 사람이 안 보이면 상태 초기화
        self.coherence = coherence          # 발이 안 보일 때: 머리/엉덩이 상승량이 어깨 상승량의 이 비율 이상이어야 점프
        self.feet_lift = feet_lift          # 발이 보일 때: 한 발이라도 몸통 길이의 이 비율 이상 올라가야 점프
        self.max_shift = max_shift          # 최저점->최고점 좌우 이동이 몸통 길이의 이 비율을 넘으면 걷기
        self.max_scale_change = max_scale_change  # 몸통 길이가 이 비율 넘게 변하면 앞뒤로 움직인 것
        self.max_width_change = max_width_change  # 엉덩이가 안 보일 때 어깨 너비 기준 (관절이 더 튀므로 느슨하게)
        self.rhythm_min = 1 if lenient else rhythm_min   # 세기 시작하는 데 필요한 연속 후보 수
        self.period_range = period_range    # 점프 간격 허용 범위 (초): 분당 40 ~ 240회
        self.period_tolerance = period_tolerance  # 연속 간격끼리 허용하는 편차 비율
        self.rhythm_break = rhythm_break    # 이보다 오래 쉬면 리듬을 다시 확인
        self.miss_correction = miss_correction and not lenient  # 쉬었다 재개하면 직전 점프를 걸린 것으로 보고 뺀다
        self.lenient = lenient              # True면 발/머리 대조, 제자리, 리듬 검사를 모두 끈다
        self.debug = debug                  # True면 후보마다 판정 내용을 log에 남긴다
        self.history = deque()              # (t, 높이) 그래프용
        self.log = deque(maxlen=200)        # 디버그 메시지
        self._history_seconds = history_seconds
        self.reset()

    # ----------------------------------------------------------------- 상태
    def reset(self):
        self.count = 0
        self.misses = 0              # 줄에 걸린 횟수 (리듬이 끊겼다 재개된 횟수)
        self.streak = 0              # 현재 연속 성공 횟수
        self.best_streak = 0         # 최고 연속 성공 횟수
        self.rejected = {"feet": 0, "motion": 0, "rhythm": 0}   # 걸러낸 이유별 횟수
        self.last_reject: Optional[tuple] = None   # (시각, 이유 문자열) 화면 표시용
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
        self._track = deque()         # (t, 머리, 엉덩이, 왼발, 오른발, x, 어깨 너비, 몸통 길이) 최근 3초
        self.in_rhythm = False        # 리듬이 확인되어 바로바로 세는 중인지
        self.pending: list[JumpEvent] = []   # 리듬 확인 전 후보 점프

    # ----------------------------------------------------------------- 갱신
    def update(self, t: float, sig: Optional[BodySignal]) -> list[JumpEvent]:
        """t: 초 단위 시각(단조 증가). sig: extract_signal 결과 또는 None(사람 없음).
        이번 프레임에 확정된 점프 목록을 돌려준다 (리듬이 잡히는 순간에는 여러 개)."""
        self.last_t = t
        if sig is None:
            self.tracking = False
            if self._last_seen_t is not None and t - self._last_seen_t > self.lost_timeout:
                self._reset_tracking()
            return []
        self.tracking = True
        self._last_seen_t = t

        # 기준 길이는 천천히 따라가게 (프레임마다 흔들리지 않도록)
        if self._scale is None:
            self._scale = sig.scale
        else:
            self._scale += 0.05 * (sig.scale - self._scale)
        if self._scale < 1e-3:
            return []

        v = -sig.center_y  # 위쪽이 양수가 되도록 뒤집는다 (픽셀)
        if self._smooth is None:
            self._smooth = v
        else:
            self._smooth += self.smoothing * (v - self._smooth)
        v = self._smooth

        # 머리/엉덩이/발 높이, 좌우 위치, 몸 크기 기록 (피크 확정 시 대조, 높이는 위쪽이 양수)
        self._track.append((t, None if sig.head_y is None else -sig.head_y,
                            None if sig.hip_y is None else -sig.hip_y,
                            None if sig.feet_y is None else -sig.feet_y[0],
                            None if sig.feet_y is None else -sig.feet_y[1],
                            sig.center_x, sig.width, sig.torso))
        while self._track and t - self._track[0][0] > 3.0:
            self._track.popleft()

        # 그래프용 기록 (몸통 길이 단위)
        self.history.append((t, v / self._scale))
        while self.history and t - self.history[0][0] > self._history_seconds:
            self.history.popleft()

        if self._valley_val is not None:
            self.height = (v - self._valley_val) / self._scale
        return self._step(t, v)

    def _step(self, t: float, v: float) -> list[JumpEvent]:
        thr = self.threshold * self._scale
        if self._state == "init":
            self._state = "falling"     # 먼저 최저점(서 있는 자세)을 찾는다
            self._ext_val, self._ext_t = v, t
            return []

        if self._state == "rising":
            if v >= self._ext_val:
                self._ext_val, self._ext_t = v, t
            elif self._ext_val - v >= thr:          # 최고점에서 thr 만큼 내려옴 -> 피크 확정
                events = self._on_peak(self._ext_t, self._ext_val)
                self._state = "falling"
                self._ext_val, self._ext_t = v, t
                return events
        else:  # falling
            if v <= self._ext_val:
                self._ext_val, self._ext_t = v, t
            elif v - self._ext_val >= thr:          # 최저점에서 thr 만큼 올라옴 -> 밸리 확정
                self._valley_val, self._valley_t = self._ext_val, self._ext_t
                self._state = "rising"
                self._ext_val, self._ext_t = v, t
        return []

    def _reject(self, t: float, key: str, bucket: Optional[str], detail: str = "") -> list:
        self.last_reject = (t, REASONS[key])
        if bucket:
            self.rejected[bucket] += 1
        if self.debug:
            self.log.append(f"[{t:7.2f}s] 후보 거부: {REASONS[key]} {detail}".rstrip())
        return []

    def _on_peak(self, peak_t: float, peak_val: float) -> list[JumpEvent]:
        if self._valley_val is None:
            return []
        amplitude = (peak_val - self._valley_val) / self._scale
        rise_time = peak_t - self._valley_t
        detail = f"(amp {amplitude:.2f}, rise {rise_time:.2f}s)"
        if amplitude > self.max_amplitude:      # 추적이 튄 것
            return self._reject(peak_t, "big", None, detail)
        if rise_time > self.max_rise_time:      # 천천히 일어난 것 (점프 아님)
            return self._reject(peak_t, "slow", None, detail)
        if self._last_peak_t is not None and peak_t - self._last_peak_t < self.min_interval:
            return self._reject(peak_t, "soon", None, detail)
        if not self.lenient:
            key = self._coherent(self._valley_t, peak_t, peak_val - self._valley_val)
            if key:
                return self._reject(peak_t, key, "feet", detail)
            key = self._in_place(self._valley_t, peak_t)
            if key:
                return self._reject(peak_t, key, "motion", detail)
        self._last_peak_t = peak_t
        candidate = JumpEvent(index=0, t=peak_t, amplitude=amplitude, rise_time=rise_time)
        if self.debug:
            self.log.append(f"[{peak_t:7.2f}s] 점프 후보 {detail}")
        return self._rhythm(candidate)

    def _sample_at(self, t: float):
        best = None
        for s in self._track:
            if best is None or abs(s[0] - t) < abs(best[0] - t):
                best = s
        return best

    def _coherent(self, t_valley: float, t_peak: float, body_rise: float) -> Optional[str]:
        """최저점->최고점 사이에 몸이 실제로 떴는지. 통과하면 None, 아니면 거부 이유 키.
        발목이 보이면: 한 발이라도 몸통 길이의 feet_lift 이상 올라가야 한다 (발이 붙어 있으면 거부).
        몸통은 무릎 굽힘까지 포함해 오르내리므로 발 상승량을 어깨 상승량과 비율로 비교하지 않는다.
        발이 안 보이면: 머리 또는 엉덩이가 어깨 상승량의 coherence 비율 이상 같이 올라가야 한다.
        아무것도 안 보이면 통과."""
        a, b = self._sample_at(t_valley), self._sample_at(t_peak)
        if a is None or b is None:
            return None
        if None not in (a[3], a[4], b[3], b[4]):
            lift = max(b[3] - a[3], b[4] - a[4])
            return None if lift >= self.feet_lift * self._scale else "feet"
        rises = [b[i] - a[i] for i in (1, 2) if a[i] is not None and b[i] is not None]
        if not rises:
            return None
        return None if max(rises) >= self.coherence * body_rise else "head"

    def _median_at(self, t: float, idx: int, win: float = 0.04) -> Optional[float]:
        """t 주변 ±win 초(약 3프레임) 샘플의 idx 항목 중앙값. 한 프레임 튄 값을 걸러내되
        꾸준한 변화는 그대로 반영한다. 값이 없으면 None."""
        vals = sorted(s[idx] for s in self._track if abs(s[0] - t) <= win and s[idx] is not None)
        if not vals:
            return None
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    def _in_place(self, t_valley: float, t_peak: float) -> Optional[str]:
        """제자리에서 뛰었는지: 좌우 이동이 작고 몸 크기(카메라와의 거리)가 유지되어야 한다.
        몸통 길이(엉덩이가 보일 때)가 어깨 너비보다 훨씬 안정적이라 우선 쓴다."""
        a, b = self._sample_at(t_valley), self._sample_at(t_peak)
        if a is None or b is None:
            return None
        if abs(b[5] - a[5]) > self.max_shift * self._scale:
            return "shift"
        ta, tb = self._median_at(t_valley, 7), self._median_at(t_peak, 7)
        if ta and tb:
            if abs(tb - ta) > self.max_scale_change * ta:
                return "scale"
            return None
        wa, wb = self._median_at(t_valley, 6), self._median_at(t_peak, 6)
        if wa and wb and abs(wb - wa) > self.max_width_change * wa:
            return "scale"
        return None

    def _ref_period(self) -> Optional[float]:
        """현재 리듬의 대표 주기: 최근 점프 간격들의 중앙값."""
        recent = self.events[-6:]
        gaps = sorted(recent[i + 1].t - recent[i].t for i in range(len(recent) - 1))
        if not gaps:
            return None
        return gaps[len(gaps) // 2]

    def _regular_suffix(self) -> list[JumpEvent]:
        """pending 뒤쪽에서 간격이 규칙적인 최대 구간을 돌려준다 (앞쪽의 불규칙한 후보는 제외)."""
        run = [self.pending[-1]]
        lo, hi = self.period_range
        for c in reversed(self.pending[:-1]):
            gaps = [run[0].t - c.t] + [run[i + 1].t - run[i].t for i in range(len(run) - 1)]
            if lo <= gaps[0] <= hi and (max(gaps) - min(gaps)) <= self.period_tolerance * max(gaps):
                run.insert(0, c)
            else:
                break
        return run

    def _rhythm(self, cand: JumpEvent) -> list[JumpEvent]:
        """리듬 검사. 세기로 확정된 점프 목록을 돌려준다."""
        t = cand.t
        if self.in_rhythm:
            gap = t - self.events[-1].t
            ref = self._ref_period()
            if gap <= self.rhythm_break and (ref is None or 0.65 * ref <= gap <= 1.5 * ref):
                return [self._count(cand)]
            self.in_rhythm = False              # 오래 쉬었거나 간격이 어긋남 -> 리듬을 다시 확인한다
            if self.debug:
                self.log.append(f"[{t:7.2f}s] 리듬 끊김 (간격 {gap:.2f}s, 주기 {ref or 0:.2f}s)")
        if self.pending and t - self.pending[-1].t > self.period_range[1]:
            self.rejected["rhythm"] += len(self.pending)   # 이어지지 못한 후보들은 버린다
            if self.debug:
                self.log.append(f"[{t:7.2f}s] 후보 {len(self.pending)}개 버림: 리듬이 이어지지 않음")
            self.pending = []
        self.pending.append(cand)
        if len(self.pending) >= self.rhythm_min:
            run = self._regular_suffix()
            if len(run) >= self.rhythm_min:
                dropped = len(self.pending) - len(run)
                if dropped:
                    self.rejected["rhythm"] += dropped        # 규칙적 구간 앞의 불규칙 후보는 버린다
                if self.events and run[0].t - self.events[-1].t > self.rhythm_break:
                    # 쉬었다가 다시 시작함: 직전 점프는 줄에 걸린 시도로 보고 뺀다
                    if self.miss_correction:
                        removed = self.events.pop()
                        self.count -= 1
                        self.misses += 1
                        if self.debug:
                            self.log.append(f"[{t:7.2f}s] 걸림으로 판단: {removed.t:.2f}s 점프 1회 제외 (실패 {self.misses})")
                    self.streak = 0
                counted = [self._count(c) for c in run]   # 앞의 후보들도 한꺼번에 반영
                self.pending = []
                self.in_rhythm = True
                if self.debug:
                    self.log.append(f"[{t:7.2f}s] 리듬 확인: {len(counted)}개 반영 (총 {self.count})")
                return counted
            if self.debug:
                gaps = [self.pending[i + 1].t - self.pending[i].t for i in range(len(self.pending) - 1)]
                self.log.append(f"[{t:7.2f}s] 리듬 대기 {len(self.pending)}/{self.rhythm_min} "
                                f"(간격 {', '.join(f'{g:.2f}' for g in gaps)})")
        elif self.debug:
            self.log.append(f"[{t:7.2f}s] 리듬 대기 {len(self.pending)}/{self.rhythm_min}")
        return []

    def _count(self, cand: JumpEvent) -> JumpEvent:
        self.count += 1
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        cand.index = self.count
        self.events.append(cand)
        return cand

    # ----------------------------------------------------------------- 조회
    @property
    def state(self) -> str:
        return self._state

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

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
