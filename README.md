# LOKI Memory

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
