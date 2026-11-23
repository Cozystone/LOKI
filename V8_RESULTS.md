# LOKI v8 · 행동 결과를 단서로 쓰는 왕복 기억

같은 장치처럼 보여도 움직여 본 결과가 다르면 서로 다른 과거를 회상할 수 있는지 시험했다. 고르는 권한은 ATANOR에 있다. 이 변경은 LOKI 연구 브랜치에서만 시험했다.

## 사전 가설과 구현

[최초 등록](V8_PROTOCOL.md) → [개발 씨앗에서 발견한 보존 문제와 수정](V8_DEVELOPMENT.md) → [봉인 평가 코드](research/v8_context.py). v7과 같은 연속 가중치 W와 역공분산 P에 한 번 관측한 과거 장면을 각인한다. 질의에는 현재 시야 x, 동작 전후의 측정 b와 c, 실제로 실행한 동작 a가 들어간다. 세 벡터의 각 8개 채널은 [0,1]로 정규화하며 동작의 중립값은 0.5이다. 이전 목표 장면은 질의 입력에 없다.

    E = (2a − 1) ⊗ (c − b)
    θ = 0.14 visual(x) Ω + 2.5 vec(E) B
    k = normalize(Q [cos θ, sin θ])
    reconstruction = clip(0.5 + k W)
    support = clip(1 − 0.1 k P kᵀ)

반대 방향으로 레버를 움직였을 때 감각 변화도 반전하면 외적의 부호는 유지된다. 이에 따라 경험한 명령의 부호가 달라도 같은 반응 관계에 접근할 수 있다. 비선형·지연된 반응에서도 이 불변성이 성립하는지는 별도 문제다. 투영 Q는 v7 이식 시 P에서 이미 강하게 학습된 최대 32개 방향을 보호한다. 시야만 들어오면 기존 v7 키를 그대로 쓴다. 장면 사본 또는 검색용 목록은 저장하지 않는다. 문턱 0.65는 v7에서 고정했다. support는 관측이 학습 공간과 얼마나 겹치는지를 나타내며 정체가 맞을 확률을 뜻하지 않는다.

## 모의 왕복 측정

보지 않은 24개 씨앗마다 외형 8개에 쌍둥이 장치 2개씩을 두었다. 두 장치의 과거 장면과 반응 벡터를 독립 추출했다. 노트북에서 쌍 중 하나를 겪고 안경에서 다른 하나를 겪었다. 모의 카메라 이동·노출·잡음, 새로운 동작 부호/세기 표본, 몸 센서 잡음을 적용했다. 각 조건에 동일한 과거와 관측을 제시했고 v7은 시야만 사용했다. 가장 가까운 과거 장면을 찾는 16지선다는 채점기에서만 쓴다. 각 방향에서 진짜 기억을 승인하고 맞힌 분모에는 거절도 포함한다.

| 항목 | v7 | v8 |
|---|---:|---:|
| 미경험 반응의 닮은 장치 오승인 | 268/384 | **16/384** |
| 안경에서 노트북 기억 식별 / 승인 정답 | 176/192 · 18/192 | **192/192 · 192/192** |
| 노트북에서 안경 기억 식별 / 승인 정답 | 0/192 · 0/192 | **192/192 · 192/192** |
| 왕복 뒤 출발 시 경험한 장치 식별 | 125/192 | **192/192** |
| 함께 이식한 기존 v7 장면 식별 | 192/192 | **192/192** |
| 새 학습 후 기존 v7 장면 복원 MSE | 0.001290 | **0.000649** |

오승인율 감소는 **65.6%p**. 씨앗 단위로 24/24 개선, 짝지은 씨앗 차이의 95% 백분위 부트스트랩 범위는 61.5~69.8%p(재표집 20,000회, 씨앗 800)였다. 모든 장치가 같은 레버를 사용하는 별도 12개 씨앗에서도 오승인은 131/192에서 7/192로 줄었고 양방향 v8 승인 정답은 각각 96/96이었다. 원래 v7의 쉬운 폐쇄형 시험은 입력에 행동 이력이 없다. 원래 절차를 그대로 재실행하면 두 구현 모두 왕복 89/96, 88/96, 출발 기억 96/96, 닮은 장치 오승인 105/192이며 12개 씨앗 각각의 점수가 같다. [새 시험 원자료](measurements/v8_context.json) · [원래 시험 재검증](measurements/v8_legacy.json)

**실패 조건.** 같은 외형의 두 장치에서 실제 행동 이력을 서로 바꿔 넣으면 v8은 384회 모두 승인하면서 384회 모두 상대 장면을 골랐다. 사건의 인과적 출처를 입증하는 기관은 아니다. 호스트는 같은 상호작용에서 실제로 잰 세 행만 묶어야 한다. 반응 측정 전에는 시각 전용 회상이 가능하지만 닮은 장치에 대해 v7의 한계가 남는다. 물체마다 비교적 안정된 즉시 선형 반응이 있고 두 몸이 같은 채널 순서와 정규화를 공유하는 합성 환경이다. 실제 안경의 다른 배선·센서 지연·시야, 장기 과부하, ATANOR 행동 개선은 미측정이다.

## 예산과 지연

필드 246,144 B, 직전 장면·몸 신호 3,104 B, 이력 96 B, 합계 **249,344 B (243.5 KiB)**. 한 개의 연속 float32 배열 안에 학습 상태와 고정 투영이 있다. 256 KiB는 수치 기억 예산이다. NumPy 임시 배열, 실행 파일, SQLite 트랜잭션 사본과 프로세스 RAM은 별도다.

| 개발 PC, µs, 중앙값 / p95 | v7 | v8 |
|---|---:|---:|
| 전체 장면 읽기 | 47.25 / 117.71 | 28.80 / 45.43 |
| 학습 1회, 커밋 제외 | 68.70 / 161.14 | 47.10 / 86.51 |
| 16박자 묶음 커밋 포함, 박자당 환산 | 419.83 / 467.04 | 363.88 / 416.68 |

같은 Python 인계 런타임에서 순서를 번갈아 측정했다. v8은 C99로 위상 계산을 합쳤고 v7은 배포된 Python/NumPy 경로다. 측정치는 이 구현 조합의 벽시계 시간이다. 양쪽을 모두 C로 구현했을 때의 우위, NumPy 대체 경로의 속도, 배터리 소모는 미측정이다.

## 사용과 재현

```python
from loki_memory import ContextMemory, Transition

# 기존 DB는 ContextMemory.resume(path, identity, assigned_body)로 인계 가능
m = ContextMemory.create(path, "laptop")
m.cut()
with m.wake() as stream:
    stream.observe(earlier_frame, earlier_body)
    stream.observe_transition(current_frame, observed_history)
image, body, support = m.recall(Transition(query_frame, query_history))
```

history는 동작 전 감각, 실행 명령, 동작 후 감각 순서의 (3,8) 배열이다. 몸을 넘길 때 기존 transfer/resume 절차를 사용한다. v8 확장 필드로 첫 쓰기를 확정한 뒤에는 ContextMemory로 재개한다.

```powershell
gcc -O3 -std=c99 -shared native/phase8.c -o loki_memory/_phase8.dll
$env:OPENBLAS_NUM_THREADS="1"
python -B -m unittest discover -q
python -B -m research.v8_context --final
python -B -m research.v8_legacy
```

Linux에서는 C99 라이브러리를 `cc -O3 -std=c99 -shared -fPIC native/phase8.c -lm -o loki_memory/_phase8.so`로 빌드할 수 있다. 라이브러리가 없으면 NumPy로 동작한다. Claude의 A1과 ATANOR v4 다리는 독립적으로 진행할 수 있다.
## 실시간 몸 인계 시 동작의 출처 보존

노트북에서 시작한 동작의 결과가 안경에 늦게 도착하는 경우를 24개 새 씨앗으로 재현했다. 기존 ContextMemory의 외부 이력 입력은 안경에 이전 관측이 없으면 쓰지 않았지만, 첫 관측이 들어온 뒤에는 24/24에서 잘못된 이력으로 필드를 바꿨다. 새 실시간 접면 BodyMemory는 begin_action에서 해당 몸의 직전 장면·감각·실행 명령을 복사하고, 같은 세션의 미완료 동작과 일치하는 콜백을 한 번만 받아들인다. 늦은 콜백은 새 몸의 진행 중인 동작도 소비하지 않는다.

- 잘못된 쓰기 0/24. 인계 직전 끝난 경험은 24/24에서 이미지·몸 신호·지지 점수와 전체 필드가 인계 직후 비트 단위로 일치했다.
- 안경에서 새 경험을 하나 학습하고 돌아온 뒤에도 정상 경험 두 개만 학습한 기준 실행과 전체 필드 및 두 회상 출력이 24/24 일치했다. 추가 학습이 있는 경우 비교 기준은 그 학습을 동일하게 받은 실행이다.
- 같은 몸에서 재시작, 중복 호출, cut, 실패한 작업 재시도, 쓰기 도중 프로세스 강제 종료도 검사했다.
- 수치 기억은 249,280 B (필드 246,144 B, 진행 중인 장면·감각·명령 3,136 B). Python 객체, SQLite, 일시 계산 배열은 별도다. 개발 PC에서 16동작씩 커밋한 박자당 환산 시간(중앙값/p95)은 기존 1.253/1.764 ms, 새 경로 1.259/1.692 ms였다. 동일 스트림의 최종 필드는 같았다. 실물 전력과 지연은 미측정이다.

실시간 몸 연결 예시. 명령을 실제 실행한 쪽에서 ticket을 캡처해 해당 완료 콜백에 전달한다.

```python
from loki_memory import BodyMemory

laptop = BodyMemory.resume(path, identity, "laptop")
ticket = laptop.begin_action(before_frame, before_signals, executed_command)
# 같은 몸의 결과가 도착하면 ticket을 한 번 사용한다.
written = laptop.finish_action(ticket, after_frame, after_signals)

# 다른 동작의 결과를 기다리는 중 인계
late = laptop.begin_action(frame, signals, next_command)
glasses = BodyMemory.resume(path, laptop.transfer("glasses"), "glasses")
assert not glasses.finish_action(late, first_glasses_frame, first_glasses_signals)
```

여러 박자를 묶을 때는 `with memory.wake() as w` 안에서 같은 두 메서드를 쓴다. 서로 다른 wake 사이에도 커밋한 미완료 동작은 그 프로세스와 세션에서만 유효하다. cut·인계·재개로 미완료 동작을 버린다. 저장 파일은 기존 v8과 호환된다. 외부에서 조립한 이력을 받는 ContextMemory는 비교·오프라인 처리에 남아 있으므로 실제 몸의 쓰기는 BodyMemory로 연결해야 한다. 호스트가 새 ticket에 엉뚱한 센서 표본을 직접 붙이는 경우까지 감별하지는 않는다. ATANOR 다리는 변경하지 않았다.

[시험 전 가설](V8_HANDOFF_PROTOCOL.md) · [씨앗별 결과](measurements/v8_handoff.json) · [지연 측정](measurements/v8_handoff_latency.json). 재현 명령은 `python -B -m research.v8_handoff`, `python -B -m research.v8_handoff_latency`, `python -B -m unittest discover -q`이다.
