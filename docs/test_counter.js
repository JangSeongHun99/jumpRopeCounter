/* test_counter.js - jump_counter.js 단위 테스트 (tools/test_counter.py 와 같은 합성 케이스).
 *   node docs/test_counter.js
 */
"use strict";
const { JumpCounter, extractSignal } = require("./jump_counter.js");

const W = 1280, H = 720, FPS = 30;

function lm(x, y, v) { return { x, y, visibility: v }; }

function makePose(o) {
  o = o || {};
  const shoulderDy = o.shoulderDy || 0, bodyDy = o.bodyDy || 0, feetDy = o.feetDy || 0;
  const hipsVisible = o.hipsVisible !== false, feetVisible = o.feetVisible !== false;
  const zoom = o.zoom || 0, xOff = o.xOff || 0, hipsY = o.hipsY === undefined ? 0.62 : o.hipsY;
  const pts = Array.from({ length: 33 }, () => lm(0.5, 0.5, 0.0));
  const put = (i, x, y, v, dy) => {
    v = v === undefined ? 0.95 : v; dy = dy === undefined ? bodyDy : dy;
    pts[i] = lm(0.5 + (x - 0.5) * (1 + zoom) + xOff, 0.5 + (y - dy - 0.5) * (1 + zoom), v);
  };
  put(0, 0.50, 0.25); put(7, 0.47, 0.26); put(8, 0.53, 0.26);
  put(11, 0.42, 0.40 - shoulderDy); put(12, 0.58, 0.40 - shoulderDy);
  const hv = hipsVisible ? 0.95 : 0.1;
  put(23, 0.45, hipsY, hv); put(24, 0.55, hipsY, hv);
  const fv = feetVisible ? 0.9 : 0.1;
  put(27, 0.45, 0.95, fv, feetDy); put(28, 0.55, 0.95, fv, feetDy);
  return pts;
}

function bumps(n, amp, kind, o) {
  o = o || {};
  const periodS = o.periodS || 0.8, bumpS = o.bumpS || 0.4, driftX = o.driftX || 0, zoomAmp = o.zoomAmp || 0;
  const tailS = o.tailS === undefined ? 1.0 : o.tailS, hipsEdge = !!o.hipsEdge;
  const frames = [];
  const total = Math.floor(n * periodS * FPS) + Math.floor(tailS * FPS);
  for (let i = 0; i < total; i++) {
    const t = i / FPS, k = Math.floor(t / periodS), phase = t - k * periodS;
    const active = k < n && phase < bumpS;
    const d = active ? amp * Math.sin(Math.PI * phase / bumpS) : 0.0;
    const pose = makePose({
      shoulderDy: kind === "shrug" ? d : 0, bodyDy: ["jump", "squat", "lean"].includes(kind) ? d : 0,
      feetDy: kind === "jump" ? d : 0, hipsVisible: o.hipsVisible, feetVisible: o.feetVisible,
      zoom: (active && amp) ? d / amp * zoomAmp : 0, xOff: driftX * Math.min(t / periodS, n),
      hipsY: hipsEdge ? (i % 2 === 0 ? 0.99 : 1.04) : 0.62,
    });
    if (hipsEdge) { pose[23].visibility = 0.6; pose[24].visibility = 0.6; }
    frames.push(pose);
  }
  return frames;
}

function run(seq, opts) {
  const c = new JumpCounter(opts);
  seq.forEach((p, i) => c.update(i / FPS, extractSignal(p, W, H)));
  return c;
}

const caught = bumps(5, 0.04, "jump", { tailS: 0 }).slice(0, Math.floor((4 * 0.8 + 0.4) * FPS))
  .concat(bumps(1, 0.03, "jump", { periodS: 0.25, bumpS: 0.25, tailS: 3.0 }), bumps(5, 0.04, "jump"));
const quick = bumps(5, 0.04, "jump", { tailS: 0 }).slice(0, Math.floor((4 * 0.8 + 0.4) * FPS))
  .concat(bumps(1, 0.03, "jump", { periodS: 0.25, bumpS: 0.25, tailS: 1.3 }), bumps(5, 0.04, "jump"));
const stutter = bumps(3, 0.04, "jump", { tailS: 0.6 }).concat(bumps(3, 0.04, "jump"));
const rested = bumps(5, 0.04, "jump", { tailS: 2.5 }).concat(bumps(5, 0.04, "jump"));
const tempo = bumps(5, 0.05, "jump", { tailS: 0 }).concat(bumps(8, 0.03, "jump", { periodS: 0.4, bumpS: 0.24 }));

const cases = [
  ["점프 5회 (전신 보임)", bumps(5, 0.04, "jump"), 5],
  ["점프 5회 (발 안 보임)", bumps(5, 0.04, "jump", { feetVisible: false }), 5],
  ["점프 5회 (상반신만 보임)", bumps(5, 0.04, "jump", { hipsVisible: false, feetVisible: false }), 5],
  ["작은 점프 5회 (진폭 몸통의 7%)", bumps(5, 0.015, "jump"), 5],
  ["빠른 점프 12회 (분당 200회)", bumps(12, 0.03, "jump", { periodS: 0.3, bumpS: 0.24 }), 12],
  ["점프 5회 (엉덩이가 화면 가장자리, 발 안 보임)", bumps(5, 0.04, "jump", { feetVisible: false, hipsEdge: true }), 5],
  ["느린 점프 5회 (분당 50회)", bumps(5, 0.05, "jump", { periodS: 1.2, bumpS: 0.4 }), 5],
  ["점프 3회 (리듬 확인 최소 횟수)", bumps(3, 0.04, "jump"), 3],
  ["점프 2회만 (리듬 미확인)", bumps(2, 0.04, "jump"), 0],
  ["한 번 튕기기", bumps(1, 0.06, "jump"), 0],
  ["어깨 으쓱 5회 (전신 보임)", bumps(5, 0.04, "shrug"), 0],
  ["어깨 으쓱 5회 (상반신만 보임)", bumps(5, 0.04, "shrug", { hipsVisible: false, feetVisible: false }), 0],
  ["발 붙이고 무릎만 굽혔다 펴기 5회", bumps(5, 0.04, "squat"), 0],
  ["앉아서 앞뒤로 흔들기 5회 (엉덩이 보임)", bumps(5, 0.04, "lean", { feetVisible: false, zoomAmp: 0.25 }), 0],
  ["앉아서 앞뒤로 크게 흔들기 5회 (상반신만)", bumps(5, 0.04, "lean", { hipsVisible: false, feetVisible: false, zoomAmp: 0.5 }), 0],
  ["걷기 5걸음", bumps(5, 0.04, "jump", { driftX: 0.3 }), 0],
  ["가만히 있기", bumps(0, 0.0, "jump"), 0],
  ["줄에 걸림: 5회 -> 걸림+허둥댐 -> 3초 -> 5회", caught, 9, { misses: 1, bestStreak: 5 }],
  ["빠른 재개: 5회 -> 걸림 -> 1.3초 -> 5회", quick, 9, { misses: 1, bestStreak: 5 }],
  ["박자 하나 멈칫 (간격 1.4초): 3회 -> 3회", stutter, 6, { misses: 0, bestStreak: 6 }],
  ["쉬었다 재개: 5회 -> 2.5초 -> 5회 (직전 1회 빠짐)", rested, 9, { misses: 1, bestStreak: 5 }],
  ["템포 변경: 느리게 5회 -> 바로 빠르게 8회", tempo, 13, { misses: 0, bestStreak: 13 }],
];

let ok = true;
for (const [name, seq, expect, extra] of cases) {
  const c = run(seq);
  let passed = c.count === expect;
  for (const [k, v] of Object.entries(extra || {})) passed = passed && c[k] === v;
  ok = ok && passed;
  console.log(`[${passed ? "OK" : "FAIL"}] ${name}: ${c.count}회 (기대 ${expect}; 걸러냄 발 ${c.rejected.feet}, 이동 ${c.rejected.motion}, 리듬 ${c.rejected.rhythm}, 대기 ${c.pending.length}; 걸림 ${c.misses}, 최고 연속 ${c.bestStreak})`);
}
const lenient = run(bumps(2, 0.04, "shrug"), { lenient: true });
const lp = lenient.count === 2;
ok = ok && lp;
console.log(`[${lp ? "OK" : "FAIL"}] lenient 모드: 어깨 으쓱 2회도 셈: ${lenient.count}회 (기대 2)`);
console.log(ok ? "모두 통과" : "실패 있음");
process.exit(ok ? 0 : 1);
