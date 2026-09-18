/* jump_counter.js - 줄넘기 점프 판정 로직 (jump_counter.py 를 그대로 옮긴 것).
 * 브라우저(<script>)와 Node(require) 양쪽에서 쓸 수 있다. 판정 규칙 설명은 jump_counter.py 참고.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.JumpCounterLib = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // MediaPipe Pose 33개 랜드마크 중 사용하는 인덱스
  const NOSE = 0, L_EAR = 7, R_EAR = 8, L_SHOULDER = 11, R_SHOULDER = 12,
        L_HIP = 23, R_HIP = 24, L_ANKLE = 27, R_ANKLE = 28;

  // 거부 이유 (화면 표시용)
  const REASONS = {
    feet: "발이 뜨지 않음",
    head: "머리·엉덩이가 같이 올라가지 않음",
    shift: "옆으로 이동함",
    scale: "카메라 쪽으로 다가가거나 멀어짐",
    slow: "너무 느리게 올라감",
    soon: "직전 점프와 너무 가까움",
    big: "추적이 튐",
  };

  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
  function vis(lm) { return (lm.visibility === undefined || lm.visibility === null) ? 1.0 : +lm.visibility; }
  function median(arr) {
    const v = arr.slice().sort((a, b) => a - b);
    const n = v.length;
    if (!n) return null;
    return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
  }

  /** 정규화 랜드마크 목록 -> 점프 판정용 신호 (픽셀 단위). 어깨·엉덩이 둘 다 안 보이면 null. */
  function extractSignal(landmarks, width, height, opts) {
    opts = opts || {};
    const minVis = opts.minVisibility === undefined ? 0.5 : opts.minVisibility;
    const source = opts.source || "torso";
    const feetVis = opts.feetVisibility === undefined ? 0.7 : opts.feetVisibility;
    const pt = (i) => {
      const lm = landmarks[i];
      let v = vis(lm);
      if (!(lm.x >= -0.02 && lm.x <= 1.02 && lm.y >= -0.02 && lm.y <= 1.02)) v = 0.0;  // 화면 밖 추정은 무시
      return [lm.x * width, lm.y * height, v];
    };
    const ls = pt(L_SHOULDER), rs = pt(R_SHOULDER), lh = pt(L_HIP), rh = pt(R_HIP);
    const shOk = ls[2] >= minVis && rs[2] >= minVis;
    const hipOk = lh[2] >= minVis && rh[2] >= minVis;
    if (!shOk && !hipOk) return null;

    const shMid = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
    const hipMid = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2];
    const shoulderW = shOk ? dist(ls, rs) : null;
    let scale;
    if (shOk && hipOk) scale = dist(shMid, hipMid);
    else if (shOk) scale = shoulderW * 1.25;           // 상반신만 보임: 어깨 너비로 추정
    else scale = dist(lh, rh) * 2.5;                   // 엉덩이만 보임 (드묾)
    if (scale < 4) return null;
    const center = shOk ? shMid : hipMid;

    const la = pt(L_ANKLE), ra = pt(R_ANKLE);
    const feetOk = la[2] >= feetVis && ra[2] >= feetVis;
    let centerY = center[1];
    if (source === "feet" && la[2] >= minVis && ra[2] >= minVis) centerY = (la[1] + ra[1]) / 2;

    const headPts = [pt(NOSE), pt(L_EAR), pt(R_EAR)].filter((p) => p[2] >= minVis);
    const headY = headPts.length ? headPts.reduce((s, p) => s + p[1], 0) / headPts.length : null;
    return {
      centerY, scale, centerX: center[0], width: shoulderW,
      torso: (shOk && hipOk) ? scale : null,
      headY, hipY: hipOk ? hipMid[1] : null,
      feetY: feetOk ? [la[1], ra[1]] : null,
    };
  }

  class JumpCounter {
    constructor(o) {
      o = o || {};
      const d = (k, v) => (o[k] === undefined ? v : o[k]);
      this.threshold = d("threshold", 0.05);
      this.minInterval = d("minInterval", 0.20);
      this.maxRiseTime = d("maxRiseTime", 0.5);
      this.maxAmplitude = d("maxAmplitude", 1.5);
      this.smoothing = d("smoothing", 0.5);
      this.lostTimeout = d("lostTimeout", 0.5);
      this.coherence = d("coherence", 0.5);
      this.feetLift = d("feetLift", 0.03);
      this.maxShift = d("maxShift", 0.5);
      this.maxScaleChange = d("maxScaleChange", 0.2);
      this.maxWidthChange = d("maxWidthChange", 0.4);
      this.lenient = !!d("lenient", false);
      this.rhythmMin = this.lenient ? 1 : d("rhythmMin", 3);
      this.periodRange = d("periodRange", [0.25, 1.5]);
      this.periodTolerance = d("periodTolerance", 0.5);
      this.rhythmBreak = d("rhythmBreak", 2.0);
      this.missCorrection = d("missCorrection", true) && !this.lenient;
      this.missGap = d("missGap", 1.2);
      this.debug = !!d("debug", false);
      this.historySeconds = d("historySeconds", 12.0);
      this.history = [];
      this.log = [];
      this.reset();
    }

    reset() {
      this.count = 0; this.misses = 0; this.streak = 0; this.bestStreak = 0;
      this.rejected = { feet: 0, motion: 0, rhythm: 0 };
      this.lastReject = null;
      this.events = [];
      this.lastT = null;
      this.height = 0.0;
      this.tracking = false;
      this.history.length = 0;
      this._resetTracking();
    }

    _resetTracking() {
      this._state = "init";
      this._extVal = null; this._extT = null;
      this._valleyVal = null; this._valleyT = null;
      this._lastPeakT = null;
      this._smooth = null; this._scale = null; this._lastSeenT = null;
      this._track = [];
      this.inRhythm = false;
      this.pending = [];
    }

    /** t: 초, sig: extractSignal 결과 또는 null. 이번 프레임에 확정된 점프 목록을 돌려준다. */
    update(t, sig) {
      this.lastT = t;
      if (!sig) {
        this.tracking = false;
        if (this._lastSeenT !== null && t - this._lastSeenT > this.lostTimeout) this._resetTracking();
        return [];
      }
      this.tracking = true;
      this._lastSeenT = t;
      if (this._scale === null) this._scale = sig.scale;
      else this._scale += 0.05 * (sig.scale - this._scale);
      if (this._scale < 1e-3) return [];

      let v = -sig.centerY;
      if (this._smooth === null) this._smooth = v;
      else this._smooth += this.smoothing * (v - this._smooth);
      v = this._smooth;

      this._track.push([t, sig.headY === null ? null : -sig.headY,
                        sig.hipY === null ? null : -sig.hipY,
                        sig.feetY === null ? null : -sig.feetY[0],
                        sig.feetY === null ? null : -sig.feetY[1],
                        sig.centerX, sig.width, sig.torso]);
      while (this._track.length && t - this._track[0][0] > 3.0) this._track.shift();

      this.history.push([t, v / this._scale]);
      while (this.history.length && t - this.history[0][0] > this.historySeconds) this.history.shift();

      if (this._valleyVal !== null) this.height = (v - this._valleyVal) / this._scale;
      return this._step(t, v);
    }

    _step(t, v) {
      const thr = this.threshold * this._scale;
      if (this._state === "init") {
        this._state = "falling";
        this._extVal = v; this._extT = t;
        return [];
      }
      if (this._state === "rising") {
        if (v >= this._extVal) { this._extVal = v; this._extT = t; }
        else if (this._extVal - v >= thr) {
          const events = this._onPeak(this._extT, this._extVal);
          this._state = "falling";
          this._extVal = v; this._extT = t;
          return events;
        }
      } else {
        if (v <= this._extVal) { this._extVal = v; this._extT = t; }
        else if (v - this._extVal >= thr) {
          this._valleyVal = this._extVal; this._valleyT = this._extT;
          this._state = "rising";
          this._extVal = v; this._extT = t;
        }
      }
      return [];
    }

    _reject(t, key, bucket, detail) {
      this.lastReject = [t, REASONS[key], key];
      if (bucket) this.rejected[bucket] += 1;
      if (this.debug) this.log.push(`[${t.toFixed(2)}s] 후보 거부: ${REASONS[key]} ${detail || ""}`.trim());
      return [];
    }

    _onPeak(peakT, peakVal) {
      if (this._valleyVal === null) return [];
      const amplitude = (peakVal - this._valleyVal) / this._scale;
      const riseTime = peakT - this._valleyT;
      const detail = `(amp ${amplitude.toFixed(2)}, rise ${riseTime.toFixed(2)}s)`;
      if (amplitude > this.maxAmplitude) return this._reject(peakT, "big", null, detail);
      if (riseTime > this.maxRiseTime) return this._reject(peakT, "slow", null, detail);
      if (this._lastPeakT !== null && peakT - this._lastPeakT < this.minInterval) return this._reject(peakT, "soon", null, detail);
      if (!this.lenient) {
        let key = this._coherent(this._valleyT, peakT, peakVal - this._valleyVal);
        if (key) return this._reject(peakT, key, "feet", detail);
        key = this._inPlace(this._valleyT, peakT);
        if (key) return this._reject(peakT, key, "motion", detail);
      }
      this._lastPeakT = peakT;
      const cand = { index: 0, t: peakT, amplitude, riseTime };
      if (this.debug) this.log.push(`[${peakT.toFixed(2)}s] 점프 후보 ${detail}`);
      return this._rhythm(cand);
    }

    _sampleAt(t) {
      let best = null;
      for (const s of this._track) if (best === null || Math.abs(s[0] - t) < Math.abs(best[0] - t)) best = s;
      return best;
    }

    _coherent(tValley, tPeak, bodyRise) {
      const a = this._sampleAt(tValley), b = this._sampleAt(tPeak);
      if (!a || !b) return null;
      if (a[3] !== null && a[4] !== null && b[3] !== null && b[4] !== null) {
        const lift = Math.max(b[3] - a[3], b[4] - a[4]);
        return lift >= this.feetLift * this._scale ? null : "feet";
      }
      const rises = [];
      for (const i of [1, 2]) if (a[i] !== null && b[i] !== null) rises.push(b[i] - a[i]);
      if (!rises.length) return null;
      return Math.max(...rises) >= this.coherence * bodyRise ? null : "head";
    }

    _medianAt(t, idx, win) {
      win = win === undefined ? 0.04 : win;
      const vals = [];
      for (const s of this._track) if (Math.abs(s[0] - t) <= win && s[idx] !== null && s[idx] !== undefined) vals.push(s[idx]);
      return vals.length ? median(vals) : null;
    }

    _inPlace(tValley, tPeak) {
      const a = this._sampleAt(tValley), b = this._sampleAt(tPeak);
      if (!a || !b) return null;
      if (Math.abs(b[5] - a[5]) > this.maxShift * this._scale) return "shift";
      const ta = this._medianAt(tValley, 7), tb = this._medianAt(tPeak, 7);
      if (ta && tb) return Math.abs(tb - ta) > this.maxScaleChange * ta ? "scale" : null;
      const wa = this._medianAt(tValley, 6), wb = this._medianAt(tPeak, 6);
      if (wa && wb && Math.abs(wb - wa) > this.maxWidthChange * wa) return "scale";
      return null;
    }

    _refPeriod() {
      const recent = this.events.slice(-6);
      const gaps = [];
      for (let i = 0; i + 1 < recent.length; i++) gaps.push(recent[i + 1].t - recent[i].t);
      if (!gaps.length) return null;
      gaps.sort((x, y) => x - y);
      return gaps[Math.floor(gaps.length / 2)];
    }

    _regularSuffix() {
      const run = [this.pending[this.pending.length - 1]];
      const [lo, hi] = this.periodRange;
      for (let i = this.pending.length - 2; i >= 0; i--) {
        const c = this.pending[i];
        const gaps = [run[0].t - c.t];
        for (let j = 0; j + 1 < run.length; j++) gaps.push(run[j + 1].t - run[j].t);
        const mx = Math.max(...gaps), mn = Math.min(...gaps);
        if (lo <= gaps[0] && gaps[0] <= hi && (mx - mn) <= this.periodTolerance * mx) run.unshift(c);
        else break;
      }
      return run;
    }

    _rhythm(cand) {
      const t = cand.t;
      if (this.inRhythm) {
        const gap = t - this.events[this.events.length - 1].t;
        const ref = this._refPeriod();
        if (gap <= this.rhythmBreak && (ref === null || (0.65 * ref <= gap && gap <= 1.5 * ref))) return [this._count(cand)];
        this.inRhythm = false;
        if (this.debug) this.log.push(`[${t.toFixed(2)}s] 리듬 끊김 (간격 ${gap.toFixed(2)}s)`);
      }
      if (this.pending.length && t - this.pending[this.pending.length - 1].t > this.periodRange[1]) {
        this.rejected.rhythm += this.pending.length;
        if (this.debug) this.log.push(`[${t.toFixed(2)}s] 후보 ${this.pending.length}개 버림: 리듬이 이어지지 않음`);
        this.pending = [];
      }
      this.pending.push(cand);
      if (this.pending.length >= this.rhythmMin) {
        const run = this._regularSuffix();
        if (run.length >= this.rhythmMin) {
          const dropped = this.pending.length - run.length;
          if (dropped) this.rejected.rhythm += dropped;
          const ref = this._refPeriod();
          const missThr = ref ? Math.max(this.missGap, 1.8 * ref) : this.missGap;
          if (this.events.length && run[0].t - this.events[this.events.length - 1].t > missThr) {
            if (this.missCorrection) {
              const removed = this.events.pop();
              this.count -= 1;
              this.misses += 1;
              if (this.debug) this.log.push(`[${t.toFixed(2)}s] 걸림으로 판단: ${removed.t.toFixed(2)}s 점프 1회 제외 (실패 ${this.misses})`);
            }
            this.streak = 0;
          }
          const counted = run.map((c) => this._count(c));
          this.pending = [];
          this.inRhythm = true;
          if (this.debug) this.log.push(`[${t.toFixed(2)}s] 리듬 확인: ${counted.length}개 반영 (총 ${this.count})`);
          return counted;
        }
      }
      if (this.debug) this.log.push(`[${t.toFixed(2)}s] 리듬 대기 ${this.pending.length}/${this.rhythmMin}`);
      return [];
    }

    _count(cand) {
      this.count += 1;
      this.streak += 1;
      this.bestStreak = Math.max(this.bestStreak, this.streak);
      cand.index = this.count;
      this.events.push(cand);
      return cand;
    }

    get rejectedTotal() { return this.rejected.feet + this.rejected.motion + this.rejected.rhythm; }

    /** 최근 점프들로 계산한 분당 점프 수. 3초 이상 멈추면 0. */
    cadence(now, window) {
      window = window || 8;
      if (this.events.length < 2) return 0.0;
      now = (now === undefined || now === null) ? this.lastT : now;
      const recent = this.events.slice(-window).map((e) => e.t);
      if (now !== null && now - recent[recent.length - 1] > 3.0) return 0.0;
      const span = recent[recent.length - 1] - recent[0];
      return span > 0 ? 60.0 * (recent.length - 1) / span : 0.0;
    }
  }

  return { JumpCounter, extractSignal, REASONS };
});
