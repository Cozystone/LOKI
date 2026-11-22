# LOKI Memory v2

LOKI v2는 정규화된 감각 벡터와 직전 모터 명령을 받는 NumPy 기억기관이다. 한 번 본 항목을 다시 찾는 인터페이스를 없앴다. 단서를 주면 연속 상태가 관측 오차를 줄이는 방향으로 움직이고, 같은 가중치에서 모든 감각을 다시 합성한다. 가변 상태는 체크포인트에 그대로 담기는 float32 행렬 하나다.

**현재 검증한 범위**는 240개의 서로 다른 경험 뒤 첫 장면의 거친 특징 보존, 일부 교차 감각 복원, 고정 크기 스트리밍이다. 봉인한 ATANOR 방에서 원시 영상의 다음 장면 예측과 암전 복합 운동은 사전 기준을 통과하지 못했다. [실험 결과와 반례](V2_RESULTS.md) · [모형과 수식](V2_THEORY.md)

## 일반 감각 스트림

설치는 `python -m pip install .`이다. 본체는 CPU에서 실행되며 NumPy 외에 런타임 의존성이 없다.

```python
import numpy as np
from loki_memory import LokiMemory

field = LokiMemory(4, budget=256 * 1024)
field.observe(np.array([0.2, 0.7, 0.4, 0.3], np.float32))
field.observe(np.array([0.3, 0.6, 0.5, 0.3], np.float32), action=[0.2, -0.4])

cue = np.array([0.3, np.nan, np.nan, 0.3], np.float32)
reconstructed = field.recall(cue, np.isfinite(cue))
forecast = field.imagine([[0.2, -0.4], [0.3, -0.5]])
field.advance([0.2, -0.4])
field.save("loki-v2.npz", binding="body-A")
continued = LokiMemory.load("loki-v2.npz", binding="body-A")
```

* `observe(frame, action, visible=mask)`는 실제로 보인 성분의 잔차로 학습한다. 마스크는 원래 감각 성분당 불리언 하나다. 입력 범위는 0–1이다.
* `recall(cue, mask, actions=(), carry=True)`는 현재 문맥에서 단서와 맞는 상태를 찾아 합성한다. `carry=False`는 새 문맥에서 단서 자체의 복원 능력을 검사한다.
* `imagine(actions)`은 같은 출발점에서 여러 가상 행동열을 비교할 수 있는 읽기 전용 연산이다. `advance(action)`는 감각 차단 중 실제로 수행한 행동을 살아 있는 상태에 반영한다. `cut()`은 삶의 경계에서 현재 상태와 운동 흔적을 초기화한다.
* `sleep=False`는 재생을 끄는 실험 대조다. v1 체크포인트는 `fc2f85a`의 v1 코드로 읽을 수 있다. v2는 새로운 경험 스트림에서 학습을 시작한다.

## ATANOR 몸에 붙이기

`AtanorMemory`는 계약 차례대로 영상·벡터·오디오를 함께 받고, 모터 번호를 계약상의 연속 갈래 입력으로 변환한다. 실제 관측과 재구성을 따로 전달한다. 기존 몸이 한 번 읽은 번들을 공유하면 된다.

```python
from loki_memory.atanor import AtanorMemory

observer = AtanorMemory(B.c)
raw = B.raw()
observer.step(raw, cut=True)

# 기존 몸 루프에서 B.step(act)를 실행한 뒤
raw = B.raw()
prediction = observer.step(raw, previous_action=act)
# 장치 획득에 실패한 갈래는 raw의 해당 원소를 None으로 넘길 수 있다.
# 전체 감각이 끊어진 동안은 observer.step(None, previous_action=act)를 호출한다.
```

몸 하나에 인스턴스 하나를 둔다. 몸 배선에 묶어 저장할 때는 `observer.save(path, wiring_id)`와 `observer.restore(path, wiring_id)`에 같은 불투명 식별자를 전달한다. 센서별 범위와 시간 간격은 몸 접면의 일부다. 단일 박자의 실제 길이가 바뀌면 현재 구현의 시간척도도 달라진다. 노트북의 카메라·마이크를 메모리에서만 다루는 운용에서는 체크포인트 저장도 그 원칙에 맞춰야 한다. 이번 접속 시험은 ATANOR의 CPU 방 몸에서 실행했다.

## 재현

판정 스크립트는 본체와 분리돼 있다. 대조 실험에는 CPU PyTorch가 필요하고 프로세스 메모리 측정에는 psutil이 필요하다. LOKI의 v1 Git 이력이 있는 체크아웃에서 다음과 같이 재현한다.

```powershell
git worktree add --detach ../loki-v1-reference fc2f85a
git clone https://github.com/Cozystone/ATANOR_The-Creator.git ../atanor-reference
git -C ../atanor-reference checkout --detach dc05860ae6837bcdd4de0be34e811bf0848e444f
$env:LOKI_V1_REFERENCE = (Resolve-Path ../loki-v1-reference).Path
$env:ATANOR_REFERENCE = (Resolve-Path ../atanor-reference).Path
$env:OPENBLAS_NUM_THREADS = "1"
python -m unittest -v test_memory
python -m research.rooms
python -m research.retina
python -m research.dark
python -m research.ambiguity
python -m research.retention
python -m research.modalities
python -m research.stream
python -m research.footprint
```

결과는 `measurements/`에 기록된다. [최초 사전 기록](V2_PROTOCOL.md) · [첫 실패 뒤 동결한 수정 검정](V2_REVISION.md) · [v1 원본 보고서](history/v1/README.md)
