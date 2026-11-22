# LOKI Memory

LOKI는 감각 벡터와 그 직전에 수행한 익명 행동으로 자라는 독립형 기억기관이다. 65,536차원 감각을 받는 기본 구성은 장기 상태 261,088바이트에 경험 주소 80개, 공유 파형 32개, 주소마다 행동 전이 8개를 둔다. 일부만 다시 보아도 과거 장면을 재구성하고, 감각이 끊기면 지정한 행동열 뒤의 장면을 생성한다. 본체는 CPU와 NumPy로 실행된다.

## 무엇을 저장하는가

64개 성분을 한 타일로 묶고, 여러 경험에 반복해서 나타나는 파형을 관측에서 배운다. 각 경험에는 타일당 3바이트의 파형 번호·평균·진폭이 남는다. 파형을 빌려 쓰는 경험이 있는 동안 그 파형을 고정한다. 새 관측에 주소를 줄지는 **새 주소를 쓰면 얼마나 덜 틀릴 수 있는지**로 결정한다. 단서에서 가린 값은 검색 함수가 읽지 않는다. 경험을 교체할 때 그 주소로 향하는 옛 행동 전이도 함께 지운다. [수식, 설계 편향, 선행 연구](THEORY.md)

## 설치와 사용

```bash
python -m pip install .
```

```python
import numpy as np
from loki_memory import LokiMemory

memory = LokiMemory(width=65_536, budget=256 * 1024)
rng = np.random.default_rng(4)
before = rng.random(memory.width, dtype=np.float32)
after = 1 - before
memory.observe(before)
memory.observe(after, action=17)

visible = np.arange(memory.tiles) % 4 == 0
cue = after.copy()
cue[~np.repeat(visible, memory.tile)] = np.nan
reconstructed = memory.recall(cue, visible)
forecast = memory.imagine([17], start=memory.locate(before, visible))
memory.save("loki.npz")
continued = LokiMemory.load("loki.npz")
```

`observe`는 실제로 새로 들어온 감각을 기록한다. `imagine`은 현재 주소를 보존한다. `advance`는 관측이 끊긴 동안 실제로 실행한 동작으로 현재 주소를 옮긴다. `cut`은 재시작·이송·죽음에서 행동 사슬을 끊고 배운 기억을 유지한다. 주소 번호의 수명은 해당 경험이 퇴거할 때까지다.

## ATANOR와 연결

연결점은 `laptop_live.py`와 방 몸이 이미 사용하는 `B.raw()`이다. 어댑터는 계약에서 영상 갈래를 찾고 운동 번호를 그대로 받는다. 한 몸에 인스턴스 하나를 둔다. 현재 시야로 인코더를 두 번 실행하거나 카메라를 두 번 읽을 필요가 없다. 아래는 기존 루프에서 같은 `raw`를 나누어 쓰는 위치다. [검증 기준 ATANOR 리비전](https://github.com/Cozystone/ATANOR_The-Creator/tree/dc05860ae6837bcdd4de0be34e811bf0848e444f)

```python
from loki_memory.atanor import AtanorImageMemory

observer = AtanorImageMemory(B.c, budget=256 * 1024)
raw = B.raw()
observer.step(raw, cut=True)
xr = O(raw)

# 기존 루프에서 B.step(act)와 경과 시간 뒤에 한 번 획득한 raw
raw = B.raw()
reconstruction = observer.step(raw, previous_action=act)
xr2 = O(raw)
```

반환값은 계약 영상과 같은 HWC 모양의 0–1 실수 배열이다. 방 몸에서는 모두 0인 영상을 암전으로 해석한다. 검은색도 유효한 관측인 카메라에서는 `available`로 획득 성공 여부를 전달할 수 있다. 배선 순서가 섞이는 몸은 같은 불투명 배선 ID를 `observer.save(path, wiring_id)`와 `restore`에 넘겨야 이어진다. 노트북 카메라·마이크의 메모리 내 처리 원칙을 적용할 때는 어댑터도 메모리에서만 운영한다.

현재 검증한 연결은 기존 정책 옆에서 관측하고 재구성하는 영상 기관이다. 제안하는 ATANOR 변경은 실제 관측, LOKI의 재구성, 직전 장면을 다음 실측값과 대조할 수 있게 하는 것이다. 실제 몸의 한 걸음 예측 결과상, LOKI 출력을 곧바로 행동값이나 보상에 더할 근거는 약하다.

## 검증 결과

각 세계 또는 방의 MSE를 먼저 구한 뒤 같은 가중치로 평균했다. 감각 범위는 0–1이며 낮을수록 좋다.

| 질문 | LOKI | 대조군 | 사전 판정 |
|---|---:|---:|---|
| 65,536차원 합성 감각에서 75% 가린 타일 복원 | 0.0000114 | 타일 평균 0.006489, 원본 FIFO 0.030368 | 양쪽에 8/8 승, 통과 |
| 같은 세계에서 8개 행동 뒤의 감각 | 0.004512 | 타일 평균 0.009309, 직전 관측 0.032808 | 양쪽에 8/8 승, 통과 |
| ATANOR RoomBody 원시 시야 122,880차원, 75% 가림 | 0.002810 | 영상 축소 FIFO 0.002935, 직전 관측 0.006630 | 축소 FIFO에 4/8 승, 평균 4.3% 감소로 기준 미달 |
| 같은 RoomBody에서 다음 행동 뒤 원시 시야 | 0.006837 | 직전 관측 0.006633 | 1/8 승, 평균 3.1% 악화 |

합성 세계는 24개 장면이 12개의 국소 파형을 재사용하고 3개 행동이 장면을 옮긴다. 이 구조와 타일 경계가 LOKI의 편향에 잘 맞는다. 원본 FIFO는 같은 예산에서 전체 화면 3개를 보관한다. 독립 균등 잡음에서는 LOKI의 가림 복원 오차가 0.08957로 상수 0.5의 0.08297보다 컸다. 새 장면 240개를 연속 주입하자 초기에 한 번 본 장면의 오차가 0.00000444에서 0.03316으로 증가했다. [고차원 사전 기록](HIGH_DIM_PROTOCOL.md) · [세계별 수치와 반례](high_dim_results.json) · [원시 시야 사전 기록](BODY_PROTOCOL.md) · [방별 수치](body_results.json)

**장시간 실행.** 본체에 12개 반복 장면을 100,000회, float32 입력 기준 24.41 GiB 흘려보냈다. 상태는 261,088바이트로 유지됐고 첫 장면의 가림 복원 MSE는 0.00000243이었다. AMD Ryzen 9 9950X3D, 단일 BLAS 스레드에서 평균 쓰기 시간은 0.487 ms였다. 이 입력량은 장치가 흡수한 독립 정보량이나 무손실 압축률을 뜻하지 않는다. [10,000걸음 구간별 측정](endurance_results.json)

**메모리 경계.** 상태 바이트는 보유 배열과 제어값 예약분을 센다. 65,536차원 포화 상태에서 쓰기 1회의 추가 할당 추적 최고값은 약 1.70 MiB였다. 별도 Windows 프로세스에서 Python과 NumPy 등을 가져온 직후 RSS는 32.64 MiB, 일반 기억기관 시험 중 OS 최대 작업 집합은 39.31 MiB였다. 노트북이나 스마트글래스의 소비전력·발열 실측은 남아 있다. [프로세스별 자원 측정](footprint_results.json)

### 어둠 속 회전 실험

별도 연구 모듈 `loki_memory.recurrence.RecurrenceMemory`는 두 익명 동작의 감각 재귀를 이용해 주기를 고르고 푸리에 계수로 시야를 생성한다. 일반 기관에 이 두 동작 제약을 부과하지 않는다.

회전만 있는 ATANOR 방 12개 모두에서 학습된 주기와 두 번째 동작의 보폭은 (15,14)였다. 실제 코드의 감각 예측 기관 `Inner(h=32)`를 별도 12개 방에서 처음부터 학습하고 각 평가 방의 밝은 구간에서 추가 적응시켰다. 감각·행동 궤적은 공통이다. 암전 MSE는 순환 기관 0.00133, Inner 0.01358, 직전 영상 0.02556이었다. 최근 감각 전이를 그대로 찾는 대조군에는 7/12개 방에서만 이겼다. 암전 중 행동 순서를 섞으면 0.01779로 악화됐다. 관측하지 못한 전진이 섞이면 순환 기관 0.01408, Inner 0.01395로 우위가 사라졌다. 이는 R1의 64차원 채움 예측 정책을 재실행한 생존 결과와 다른 지표다.

순환 기관은 보유 상태 39,217바이트, Inner는 파라미터만 117,504바이트다. 각 모델은 같은 감각을 보지만 유도 편향·적응 방식·학습량은 다르다. 순환 기관의 별도 100,000걸음 시험에서 저장·복원 뒤 128개 연속 출력을 정확히 재현했다. [사전 기록](PROTOCOL.md) · [모든 방의 MSE](results.json) · [별도 장시간 결과](stress_results.json)

## 재현

본체와 단위 테스트에는 NumPy만 필요하다. ATANOR 실험에는 CPU에서 실행 가능한 PyTorch가 필요하고 프로세스 메모리 측정에는 psutil이 필요하다. 기록 환경은 Windows, Python 3.13.12, NumPy 2.4.6, PyTorch 2.11.0+cu128이며 모든 모델을 CPU로 실행했다.

```powershell
git clone https://github.com/Cozystone/ATANOR_The-Creator.git ../atanor-reference
git -C ../atanor-reference checkout --detach dc05860ae6837bcdd4de0be34e811bf0848e444f
$env:OPENBLAS_NUM_THREADS = "1"
python -m unittest -v test_memory
python high_dim_experiment.py --atanor ../atanor-reference
python body_experiment.py --atanor ../atanor-reference
python experiment.py --atanor ../atanor-reference
python endurance.py
python stress.py
python footprint.py
```
