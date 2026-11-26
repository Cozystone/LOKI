# LOKI Memory

## v12 · 한 생애에서 몸짓의 결과를 배우는 기억

140채널·90가지 몸짓 조합을 받는 몸에서 32박자 뒤 바탕층 변화를 예측했다. 무작위로 배정한 몸짓열에서 예측과 실제 변화의 상관은 **.700**이었다. 움직이는 자원의 다음 구석을 미리 찾아간 비율은 **51.9%**, 같은 세계에서 회상 순서만 섞으면 **20.3%**였다. 행렬은 **168,832바이트**로 고정됐다. 모든 채널에서 직전값을 이기는 기준과 낯선 몸에서의 확신 보정은 실패했다. 실제 ATANOR 방·MineLand 통합은 I0 채점이 남아 있다. [실험 결과](V12_RESULTS.md) · [Claude용 인터페이스와 소스 고정점](V12_DESIGN.md)

## v11 · 기억한 세계를 이용한 계획

### Demonstrated

같은 감각을 내는 여러 상태를 확률로 유지하고, 행동 결과로 현재 믿음을 고쳐 처음 가는 경로를 구성한다. 같은 경험·목표·32걸음에서 새 목표 도달은 v11 **21/24**, v10 **9/24**였다. 생성 규칙과 색을 바꾼 독립 방에서는 각각 **23/24**, **3/24**였다. 지정된 합성 Level 3 수치 기준을 두 코호트에서 충족했다. 1천만 사건까지 체크포인트는 **227,394바이트**였다. [실험 결과와 재현 명령](V11_RESULTS.md) · [수학적 모델](V11_THEORY.md) · [사전등록](V11_PROTOCOL.md)

### Failed

처음 보는 질감의 방에서 이산 상태의 높은 확신이 틀렸다. 첫 봉인 코호트에서는 길이 실제로 있었던 세 방에서 행동 예산을 소진했다. 닮은 두 물체가 같은 행동에 같은 반응까지 보이면 정체를 가를 증거가 부족하다. [실패 궤적](examples/v11_failures_16.png) · [확률 보정 측정](measurements/v11_calibration.json)

### Unknown

Rokid에서의 전체 실행 전력, ATANOR의 필요에 따른 선택 개선, 하나의 장기 생애에서 여러 세계를 연속 학습한 뒤의 보존율은 미측정이다. 1천만 사건 시험은 소수의 반복 가능한 상태로 구성했다. [메모리 측정 원자료](measurements/v11_capacity_16.json)

## v10 · 고정 용량의 장기 기억

백만 사건을 통합하는 동안 영속 파일은 289,100바이트로 유지됐고, 초반 단발 예외와 반복 법칙이 남았다. 다른 시점에서 같은 개체를 합치는 정밀도는 0.620, 새 3D 목표 도달은 7/24였다. **전체 사전 기준 실패.** [성공·실패·재현](V10_RESULTS.md) · [가설과 수식](V10_THEORY.md) · [프로토콜](V10_PROTOCOL.md)

## v9 · 시간순 사건과 행동 효과

감각 연합 코어에 시간 흔적, 선택형 로컬 사건 그래프, 행동 효과 예측을 연결했다. 수치 코어는 261,968 B이고 디스크 피질은 경험에 따라 커진다. 기존 3D 미로에서 학습한 역동작으로 여정을 되짚어 23/24회 귀환했다. 방향 반전과 내부 예측만으로 길 찾기는 실패했다. [실험·실패 조건·사용법](V9_RESULTS.md)

## v8 · 같은 외형, 다른 행동 이력

노트북과 모의 안경 사이에서 한 동작 전후의 감각 변화를 함께 쓰자, 새 합성 시험의 닮은 장치 오승인이 268/384에서 16/384로 줄었다. 두 방향 모두 진짜 기억을 192/192 승인하고 정확히 회상했다. 수치 상태는 243.5 KiB다. 행동 이력이 없는 원래 v7 시험에서는 두 버전의 점수가 같았다. [결과·실패 조건·재현 방법](V8_RESULTS.md) · [사전등록](V8_PROTOCOL.md)

실시간 몸 연결에는 `BodyMemory`를 사용한다. 노트북에서 시작한 동작이 안경의 첫 감각과 합쳐지는 쓰기를 차단하고, 인계 전에 마친 경험은 그대로 보존했다. [인계 시험·연결 예시](V8_RESULTS.md#실시간-몸-인계-시-동작의-출처-보존)

## v7 · 한 기억의 기기 간 왕복

고정 수치 상태 219.4 KiB에서 노트북과 모의 안경 사이에 한 작성자의 기억을 넘긴다. 다른 카메라 단서의 과거 장면 식별은 왕복 각 89/96, 88/96이었고 비슷한 다른 물체의 오승인율은 54.7%였다. C99 읽기·쓰기와 데스크톱 인계를 검증했다. [사용 예제·효율·실물 검증 범위](V7_RESULTS.md)

## v6 · 과거 시야를 떠올려 돌아가기

v6는 현재 감각에서 직전의 16×16 RGB 시야를 재구성한다. 코어와 회상 기관을 합쳐 252.375 KiB다. 새 1인칭 3D 미로에서 후진 가능한 몸은 실제 후보 시야를 확인하면 24/24 돌아왔지만, 내부 예측만으로는 0/24였다. ATANOR의 학습된 `room3d_ret6e.pt`로 시험했을 때도 코어 단독과 회상 결합이 모두 0/24였다. [사진 원본과 회상본](examples/v6-photo-recall.png) · [3D 미로 귀환 경로](examples/v6-maze.png) · [판정·Claude 전달문·재현](V6_RESULTS.md)

다음 코드는 한 번 본 전이에서 과거 시야를 읽는다. `previous`와 `current`는 0~1로 정규화한 16×16×3 실제 카메라 배열이고 `levels`는 같은 시점에 측정한 두 몸 채널이다.

```python
from loki_memory import ReverseMemory

memory = ReverseMemory()
memory.observe(previous, levels)
memory.observe(current, next_levels, action=1)
memory.cut()
past_image = memory.recall(current)
```

## v4 코어와 v5 장기 피질

LOKI는 몸의 감각과 행동에서 다음 변화를 배우는 로컬 기억기관이다. **v4**는 256 KiB 이하의 코어로 시간과 행동을 적분한다. **v5**는 코어에 선택적으로 연결하는 실험용 장기 피질이다. 반복되는 변화는 회귀 계수에, 한 번 겪은 큰 놀람은 공유 푸리에 장에 남긴다. 과거 프레임이나 사건별 키를 보관하지 않는다.

## v5 장기 기억 사용

96성분 감각에서 기본 피질의 수치 배열은 **1.43 MiB**다. 24,000번 쓴 상태에서 코어와 피질을 함께 읽는 데 이 개발용 CPU에서 p95 **0.76 ms**가 들었다. ATANOR 새 방 12개의 다음 시야 MSE는 v4보다 **7.9%** 낮았다. 사전에 정한 10% 기준에 미달해 공간 행동의 기본 경로는 계속 v4를 사용한다. 같은 방에서 측정한 전진 효과와 독립 사건을 많이 겹쳤을 때의 용량 한계는 [v5 결과](V5_RESULTS.md)에 있다.

```python
import numpy as np
from loki_memory import CorticalMemory, LokiMemory, LongTermCortex

memory = CorticalMemory(LokiMemory(96), LongTermCortex(96))
x = np.linspace(0.2, 0.8, 96, dtype=np.float32)
memory.observe(x)
expected = memory.predict(2)                   # 행동 전에 후보 결과를 읽는다
memory.observe(1-x, 2, integrity_change=-0.2)   # 몸이 실제 측정한 변화
memory.cortex.sleep()                          # 유휴 시간에 누적 회귀를 푼다
memory.cut()
memory.observe(x)
remembered = memory.recall_episode(2)           # 같은 단서에서 겪었던 이례적 결과
memory.save("memory.zip", binding="body-v1")
restored = CorticalMemory.load("memory.zip", binding="body-v1")
```

이 예제의 온전함 변화는 사용법을 보여주는 입력이다. 실기기에서는 몸에서 얻은 값만 전달하며, 측정값이 없으면 생략한다. 여러 행동을 비교할 때는 각 후보에 `predict`를 호출한다. 실행한 행동만 다음 관측과 함께 `observe`에 넘긴다. `advance`는 암전 한 박자, `cut`은 서로 이어지지 않는 경험의 경계를 뜻한다. 피질의 작업 위치는 RAM이고 디스크 파일은 재시작용이다.

[최초 사전등록](V5_PROTOCOL.md) · [예외 기억 가설](V5_REVISION.md) · [최종 사전등록](V5_FINAL_PROTOCOL.md) · [수학과 저장 형식](V5_THEORY.md) · [대조 실험·재현 명령](V5_RESULTS.md)

## v4 코어와 전진 효과

LOKI는 감각과 행동에서 배우는 로컬 기억기관이다. v4가 답하려는 질문은 “같은 장면에서 같은 순서로 돌되, 멈춰 있던 몇 박자에 앞으로 걸었다면 마지막에 보이는 무엇이 달라질까?”이다. 두 계획을 같은 출발 상태에 주면 각 시야 성분이 얼마나 커지거나 작아질지 부호 있는 벡터로 반환한다.

본체는 NumPy와 CPU로 실행된다. 가중치, 역공분산, 현재 감각과 운동 흔적을 포함한 지속 수치 상태는 **256 KiB 이하의 float32 행렬 하나**다. 파이썬 런타임, 입력 프레임, 계산 중 작업 메모리는 이 수치에 포함되지 않는다.

## 검증

처음 보는 ATANOR 시뮬레이션 방 12개에서 같은 출발점과 회전열을 유지한 전진·정지 48쌍을 검사했다. **20박자 뒤 최종 시야 차이 MSE는 0.014684**였다.

| 비교 대상 | v4 오차 감소 | 이긴 방 |
|---|---:|---:|
| 두 경로의 차이를 항상 0으로 예측 | 22.1% | 11/12 |
| ATANOR Inner | 23.3% | 10/12 |
| LOKI v3 | 14.3% | 10/12 |

240개 새 장면 뒤 첫 장면 복원 MSE **0.006366**, 합성 온도계의 256박자 뒤 레버 효과 MSE **0.027190**, 효과 방향 **96/96**으로 v3의 성과를 유지했다. 전진용 행렬은 **239,872바이트**다. 공간, 첫 장면, 온도 시험은 각각의 배선에서 학습한 기관으로 평가했다.

[실험 조건·방별 판정](RESULTS.md) · [결과를 보기 전의 가설](V4_PROTOCOL.md) · [수학과 구현](THEORY.md) · [v3의 결과](V3_RESULTS.md)

## 학습된 기관 사용

저장소 루트에서 `python -m pip install .`을 실행한다. 배포한 체크포인트는 학습 방 1–12에서 얻은 4행×8열 RGB 센서와 다섯 모터 명령의 배선에 맞는다. 다음 예제는 저장소에 포함된 한 번의 실제 분기를 재현한다. 추론이 끝난 뒤 채점에만 두 경로의 마지막 관측을 사용한다.

```python
import numpy as np
from loki_memory import LokiMemory

memory = LokiMemory.load("checkpoints/v4-room-grid.npz")
with np.load("examples/forward_idle.npz") as trial:
    memory.cut()
    memory.observe(trial["prefix"][0])
    for action, frame in zip(trial["observed_actions"], trial["prefix"][1:]):
        memory.observe(frame, int(action))
    a, b = trial["plans"]
    delta = memory.predict_effect(a, b)   # 최종 시야 A − 최종 시야 B
    truth = trial["endpoints"][0] - trial["endpoints"][1]
    print(delta.reshape(4, 8, 3).shape, np.mean((delta - truth) ** 2))
```

새 격자 센서는 `LokiMemory(width, grid_shape=(h,w,c))`로 시작한다. 같은 출발점에서 두 계획을 실행한 관측이 있으면 출발 문맥에서 `learn_effect(a,b,observed_a,observed_b)`를 호출한다. 쌍의 학습 전에는 두 절대 예측을 빼서 응답한다. `predict_effect`는 현재 상태를 변경하지 않는다. 몸이 실제로 수행한 행동과 다음 관측은 `observe`에 전달하고, 암전 중 실제로 진행한 한 박자는 `advance`로 반영한다. ATANOR의 전체 몸 계약 연결과 비영상 스트림 API는 [몸 연결 예제](V3_README.md)에 있다.

## 페타바이트 입력과 기억 용량

입력이 계속 들어와도 수치 행렬의 크기는 고정된다. 그러나 **1 PB에서 임의의 과거 원본을 찾아 정확히 복원하는 능력은 없다.** 256 KiB는 최대 2,097,152비트를 표현한다. 1 PB에 8×10¹⁵개의 독립 비트가 있다면 서로 다른 수많은 과거가 같은 기억 상태를 갖게 된다. 학습할 수 있는 것은 입력에 반복되는 구조와 과제에 필요한 일부 차이다.

v4에 독립 균등 난수 **16 GiB**를 실제로 입력했다. 행렬은 처음과 마지막 모두 259,992바이트였다. 마지막에 첫 입력의 75%를 가리면 복원 MSE는 0.08387, 모든 성분에 0.5를 답한 값은 0.08369였다. PB 규모 실행과 그 뒤의 기억 품질은 미검증이다. [스트림 원자료](measurements/v4_stream.json)

## 재현

Python 3.10 이상에서 실행한다. 연구용 대조군은 CPU PyTorch를 추가로 사용한다. PowerShell 기준으로 다음 명령은 정확한 비교 버전을 별도 경로에 준비한다.

```powershell
python -m pip install . torch
git worktree add --detach ../loki-v1-reference fc2f85a
git worktree add --detach ../loki-v3-reference 6d689b5
git clone https://github.com/Cozystone/ATANOR_The-Creator.git ../atanor-reference
git -C ../atanor-reference checkout --detach dc05860ae6837bcdd4de0be34e811bf0848e444f
$env:LOKI_V1_REFERENCE = (Resolve-Path ../loki-v1-reference).Path
$env:LOKI_V3_REFERENCE = (Resolve-Path ../loki-v3-reference).Path
$env:ATANOR_REFERENCE = (Resolve-Path ../atanor-reference).Path
$env:OPENBLAS_NUM_THREADS = "1"
python -B -m unittest -v test_memory
python -B -m research.v4_forward train
python -B -m research.v4_forward confirm
python -B -m research.v4_regression
python -B -m research.v4_stream
```

이전 버전의 설계와 한계는 [v3](V3_README.md), [v2](V2_README.md), [v1](history/v1/README.md)에 남아 있다.
