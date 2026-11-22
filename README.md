# LOKI Memory v3

LOKI는 정규화한 감각과 직전 행동으로 배우는 로컬 NumPy 기억기관이다. 감각이 끊기면 하나의 출발 상태에서 서로 다른 행동열을 끝까지 실행해 두 미래를 비교할 수 있다. 학습과 추론에 쓰이는 지속 수치 상태 전체를 최대 256 KiB인 float32 행렬 하나에 둔다.

새 ATANOR 방에서 20박자 혼합 암전 영상과 합성 배선의 256박자 열 개입은 사전 기준을 통과했다. 전진만 바꾼 두 경로의 최종 영상 차이, 독립적인 교차 감각 변화, 수면 보정의 5% 이득은 미달했다.

[검증 결과](RESULTS.md) · [모형과 수식](THEORY.md) · [최초 가설](V3_PROTOCOL.md) · [수정 전 등록](V3_REVISION.md) · [v2 기록](V2_RESULTS.md)

## 사용

설치는 `python -m pip install .`이다. 한 박자는 한 번의 호출이며, 두 박자 사이에 실제로 수행한 행동을 다음 관측에 붙인다.

```python
import numpy as np
from loki_memory import LokiMemory

m = LokiMemory(4)
m.observe(np.array([.4, .6, .5, .7]))
m.observe(np.array([.3, .6, .5, .65]), action=[.2, -.1, .4])

left  = m.imagine([[.2, -.1, .4]] * 256)
right = m.imagine([[.2, -.1, -.4]] * 64 + [[.2, -.1, .4]] * 192)
effect = right - left
m.advance([.2, -.1, .4])

cue = np.array([.3, np.nan, np.nan, .65])
scene = m.recall(cue, np.isfinite(cue), carry=False)
delta = np.array([-.1, np.nan, np.nan, -.05])
change = m.predict_change(delta, np.isfinite(delta))
m.save("loki-v3.npz", binding="body-A")
m = LokiMemory.load("loki-v3.npz", binding="body-A")
```

이 예시는 호출 순서를 보여 준다. 256박자 예측 시험에서는 각 배선마다 관측 전이 3,072개로 먼저 학습했다. `imagine`과 `recall`은 읽기 전용이다. `advance`는 실제 암전 중의 현재 상태를 전진시킨다. `cut()`은 에피소드 경계에서 현재 감각과 운동 흔적, 지역 시야장을 비운다. 학습 가중치는 유지한다. `sleep=False`는 동일한 기관에서 쓰기 직후의 직교 간섭 보정만 끄는 대조군이다.

## ATANOR 연결

```python
from loki_memory.atanor import AtanorMemory

organ = AtanorMemory(B.c)
organ.step(B.raw(), cut=True)
# 몸이 act를 수행한 직후, 같은 raw 번들을 공유한다.
current = organ.step(B.raw(), previous_action=act)
# 다음 행동 중 감각 전체를 잃었다면
forecast = organ.step(None, previous_action=next_act)
```

어댑터는 몸 계약의 숫자 입력과 모터 배선을 사용한다. 결측 채널은 `None`으로 보낼 수 있다. 마지막 센서가 지원되는 크기의 영상이면 계약에서 공간 인접성을 얻는다. 독립적인 격자형 센서는 `LokiMemory(width, grid_shape=(h,w,c))`로 감각 벡터의 마지막 HWC 구간을 지정할 수 있다. 몸마다 기관 하나를 두고 어댑터 체크포인트에는 고정 배선 식별자를 넘긴다. v2 체크포인트는 커밋 `0c98721`의 코드로 읽는다.

## 재현

PowerShell 기준이다. 학습 본체에는 NumPy, 비교 실험에는 CPU PyTorch, 자원 측정에는 psutil이 필요하다.

```powershell
git worktree add --detach ../loki-v1-reference fc2f85a
git worktree add --detach ../loki-v2-reference 0c98721
git clone https://github.com/Cozystone/ATANOR_The-Creator.git ../atanor-reference
git -C ../atanor-reference checkout --detach dc05860ae6837bcdd4de0be34e811bf0848e444f
$env:LOKI_V1_REFERENCE = (Resolve-Path ../loki-v1-reference).Path
$env:LOKI_V2_REFERENCE = (Resolve-Path ../loki-v2-reference).Path
$env:ATANOR_REFERENCE = (Resolve-Path ../atanor-reference).Path
$env:OPENBLAS_NUM_THREADS = "1"
python -B -m unittest -v test_memory
python -B -m research.v3_confirm dark
python -B -m research.v3_confirm rooms
python -B -m research.v3_confirm retention
python -B -m research.v3_confirm heat
python -B -m research.v3_forward
python -B -m research.v3_forward --idle
python -B -m research.v3_trace_ablation
python -B -m research.v3_pooling_floor
python -B -m research.v3_resources
```

최종 판정은 `measurements/v3_confirm_*.json`에 쓴다. 최초 v3 실험은 `e6ca0ee`에서 `research.v3_dark`, `research.v3_rooms`, `research.v3_retention`, `research.v3_heat`를 실행한다. 과거 v2 측정은 `0c98721`에서 재현한다.
