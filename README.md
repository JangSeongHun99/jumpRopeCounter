# 줄넘기 카운터 (노트북 카메라)

노트북 웹캠 앞에서 줄넘기를 하면 몇 번 넘었는지 실시간으로 세어 주는 프로그램입니다.
MediaPipe 포즈 추정으로 몸통의 위아래 움직임을 추적하고, 몸이 한 번 뜰 때마다 1회를 셉니다.

참고한 프로젝트: [AyushTCD/VisionBasedJumpRopeCounter](https://github.com/AyushTCD/VisionBasedJumpRopeCounter)
(동영상 파일 전용 Jupyter 노트북, 구 `mp.solutions` API). 이 프로젝트는 그 아이디어를
실시간 웹캠용으로 다시 만들었습니다.

- MediaPipe 1.0 Tasks API (`PoseLandmarker`) 사용, Python 3.10 ~ 3.14
- 카메라 읽기 / 포즈 추론 / 화면 그리기를 별도 스레드로 분리해 추론이 느려도 화면이 끊기지 않음
- 카메라 거리와 무관하게 동작하도록 움직임을 몸통 길이 기준으로 정규화
- 세션 기록(CSV/JSON) 저장, 정답 영상으로 정확도를 재는 평가 도구 포함

## 빠른 시작

1. 코드를 받습니다. 저장소 페이지 https://github.com/JangSeongHun99/jumpRopeCounter 에서
   `Code > Download ZIP`으로 내려받아 풀거나, 터미널에서 `git clone https://github.com/JangSeongHun99/jumpRopeCounter.git`
2. Python 3.10 이상을 설치합니다 (3.14에서 확인). 설치할 때 "Add python.exe to PATH"를 체크하세요.
3. `run.bat`을 더블클릭합니다. 처음 한 번은 가상환경을 만들고 패키지를 설치합니다
   (mediapipe 1.0.1, 약 150 MB). 포즈 모델 파일(약 15 MB)은 첫 실행 때 `models/` 폴더에 자동으로 내려받습니다.
4. 카메라에서 2~3 m 떨어져 어깨부터 엉덩이까지 보이게 섭니다. 점프하면 왼쪽 위 숫자가 올라갑니다.
5. `q`를 누르면 종료되고 `sessions/` 폴더에 기록이 저장됩니다.

터미널에서 직접 실행하려면:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python jump_rope_counter.py
```

## 실행 옵션

```
python jump_rope_counter.py                    기본 카메라(0번)
python jump_rope_counter.py --camera 1         다른 카메라
python jump_rope_counter.py --list-cameras     연결된 카메라 번호 찾기
python jump_rope_counter.py --video 영상.mp4    녹화된 영상 분석
python jump_rope_counter.py --video 영상.mp4 --no-display --verbose   창 없이 점프마다 한 줄 출력
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model lite/full/heavy` | full | 포즈 모델. lite가 가장 빠르고 heavy가 가장 느림 (카운트 정확도는 셋 다 같았음) |
| `--threshold` | 0.05 | 점프로 인정할 최소 진폭 (몸통 길이 대비 비율). 작을수록 민감 |
| `--min-interval` | 0.2 | 점프 사이 최소 간격(초). 0.2 = 분당 최대 300회 |
| `--signal torso/feet` | torso | 몸통 중심(상반신만 보여도 됨) 또는 발목(전신이 보일 때) |
| `--width`, `--height` | 1280, 720 | 카메라 요청 해상도 |
| `--infer-size` | 0 | 추론 전 긴 변을 이 크기로 축소 (0 = 원본) |
| `--no-mirror` | | 카메라 좌우반전 끄기 |
| `--record 파일.mp4` | | 스켈레톤과 카운트가 그려진 영상 저장 |
| `--save-dir` | sessions | 세션 기록 폴더 |
| `--no-save` | | 종료 시 기록을 남기지 않음 |

## 화면 단축키

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 종료 (기록 저장) |
| `r` | 카운트와 시간 초기화 |
| `space` | 일시정지 / 재개 |
| `s` | 지금까지의 기록을 바로 저장 |
| `+` / `-` | 민감도 올리기 / 내리기 (threshold 0.01씩) |
| `m` | 좌우반전 토글 |
| `d` | 스켈레톤과 그래프 표시 토글 |

화면 구성: 왼쪽 위에 점프 수, 분당 횟수, 경과 시간. 그 아래 상태 줄에 추론 시간(ms), 화면 fps,
모델, threshold. 아래에는 최근 6초의 몸통 높이 그래프와 점프 표시(초록 선),
오른쪽 끝의 흰 눈금이 threshold 크기입니다. 점프가 이 눈금보다 작게 보이면 `+`로 민감도를 올리세요.

## 동작 원리

1. MediaPipe Pose가 프레임마다 33개 관절 위치를 추정합니다.
2. 양 어깨와 양 엉덩이의 평균(몸통 중심)의 세로 위치를 신호로 씁니다. 상반신만 보이면 어깨만 씁니다.
3. 어깨 중점 ~ 엉덩이 중점 거리(몸통 길이)로 나눠 카메라와의 거리에 무관한 단위로 바꿉니다.
4. 히스테리시스 피크 검출: 최저점에서 threshold 이상 올라가면 상승, 최고점에서 threshold 이상
   내려오면 점프 1회로 확정합니다. 기준선이 없어서 사람이 앞뒤로 움직여도 잘못 세지 않습니다.
5. 0.2초보다 촘촘한 피크, 0.7초보다 느린 상승(앉았다 일어나기), 몸통 길이 1.5배가 넘는
   진폭(추적 튐)은 버립니다.
6. 점프는 머리와 엉덩이도 같이 올라가지만 어깨 으쓱이나 팔 동작은 어깨만 올라갑니다.
   머리(코·귀)나 엉덩이의 상승량이 몸통 상승량의 절반에 못 미치면 점프로 세지 않습니다.

자세한 로직은 `jump_counter.py`에 있고, MediaPipe나 OpenCV 없이 단독으로 테스트할 수 있습니다.

## 정확도

참고 저장소의 테스트 영상과 Kinovea 정답 파일로 잰 결과입니다 (`tools/evaluate.py`).
모델 lite / full / heavy 모두 같은 카운트가 나왔고, threshold 0.03 ~ 0.08 범위에서 결과가 같았습니다.

| 영상 | 정답 | 검출 | 비고 |
|---|---|---|---|
| Normal Jump 1 | 56 | 56 | |
| Normal Jump 2 | 78 | 77 | 정답 표시 3개가 0.16초 간격으로 붙어 있는 구간에서 1개 차이 |

하이니·점핑잭처럼 다른 방식으로 뛰어도 몸이 뜬 횟수를 그대로 셉니다.

## 성능

이 PC(CPU만 사용) 기준 프레임당 추론 시간: lite 13 ms, full 18 ms, heavy 65 ms.
추론은 별도 스레드에서 돌고 화면은 카메라 속도로 그려지므로, 추론이 카메라보다 느리면
오래된 프레임을 건너뛰고 최신 프레임만 처리합니다. 초당 15프레임 정도만 처리돼도 카운트는 유지됩니다.

웹캠을 30 fps 영상으로 모사한 테스트(정답 56회 영상): 715프레임 중 707~710프레임 처리, 프레임당 추론 17~19 ms,
카운트 56회. 메인 루프가 새 프레임을 폴링하지 않고 대기하도록 만들고 스레드 전환 간격을 1 ms로
줄인 결과이며, 그 전에는 같은 조건에서 추론이 90 ms까지 늘고 절반의 프레임을 건너뛰었습니다.

노트북이 느리면:

- `--model lite` (약 30% 빠름, 카운트 정확도 동일)
- `--width 640 --height 480` (카메라 전송과 화면 그리기 부담 감소)
- 시작 후 추론이 45 ms를 넘으면 콘솔에 힌트가 뜹니다.

## 세션 기록

종료하거나 `s`를 누르면 `sessions/session_날짜_시각.csv`(점프별 시각, 진폭)와
`.json`(총 횟수, 시간, 분당 평균, 사용한 설정)이 저장됩니다.

## 평가 도구

```
python tools/evaluate.py 영상.mp4                      점프 수와 시각 출력
python tools/evaluate.py 영상.mp4 --gt 정답.kva          정답과 비교 (Kinovea .kva 또는 .html)
python tools/evaluate.py 영상.mp4 --gt 정답.kva --sweep  threshold 여러 값 비교
python tools/evaluate.py 영상.mp4 --model lite full heavy
```

포즈 추론 결과를 `.cache/`에 저장해 두므로 같은 영상을 다른 설정으로 다시 평가할 때는 바로 끝납니다.
`python tools/test_counter.py`는 합성 데이터로 점프 5회는 세고 어깨 으쓱 5회는 세지 않는지 확인하는 단위 테스트입니다.

## 한계

- 줄 자체는 보지 않습니다. 줄 없이 뛰어도 세고, 더블언더는 1회로 셉니다.
- 한 사람만 추적합니다. 여러 명이 보이면 가장 확실하게 잡힌 사람을 따라갑니다.
- 화면 글자는 OpenCV 기본 폰트 제약으로 영어입니다.

## 파일

| 파일 | 역할 |
|---|---|
| `jump_rope_counter.py` | 메인 프로그램 (카메라/영상 입력, 스레드, 화면, 기록) |
| `jump_counter.py` | 신호 추출과 점프 판정 로직 (순수 파이썬) |
| `pose_backend.py` | MediaPipe 모델 관리와 추론, 스켈레톤 그리기 |
| `tools/evaluate.py` | 정확도 평가·튜닝 도구 |
| `tools/test_counter.py` | 카운터 로직 단위 테스트 (점프 vs 어깨 으쓱) |
| `models/` | 포즈 모델 파일 (`pose_landmarker_{lite,full,heavy}.task`) |
| `run.bat` | 가상환경 생성 + 실행 |
