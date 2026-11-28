# 통합된 ATANOR 몸의 움직임 후보 시험

## 물음과 판정

같은 몸짓을 시작한 뒤 몸 안의 두 채널이 어떻게 변했는지 읽고, ATANOR가 다시 고르는 박자에 그 몸짓을 후보로 올리면 레버의 배선 반전 뒤에도 도움이 되는가? 기존 ATANOR 레버 세계와 자라난 s3_e 존재를 사용했다. 기준은 따로 작동한 단계와 반전 단계에서 모두 원인 레버 몫이 오르고 총 사망이 늘지 않는 것이다. [실행 전 기준](MOTOR_INTERFACE_PROTOCOL.md)

## 두 시작의 결과

| 시작·조건 | 분리 끝 원인 몫 | 반전 끝 원인 몫 | 묶임 / 분리 / 반전 사망 | 재선택 때 후보 교체 |
|---|---:|---:|---:|---:|
| [9781 관찰 전용](measurements/motor_baseline.json), [기본 반복](measurements/motor_baseline_repeat.json) | .583 | .423 | 0 / 12 / 21 | 0 |
| [9781 선택 접면](measurements/motor_precredit.json) | .946 | .919 | 0 / 1 / 6 | 3,573 / 3,644 |
| [9782 기본](measurements/motor_9782_native.json) | .527 | .422 | 0 / 16 / 14 | 0 |
| [9782 선택 접면](measurements/motor_9782_at-choice.json) | .987 | .961 | 0 / 3 / 0 | 3,988 / 4,029 |

각 단계 마지막 3분의 1에서 실제로 접수된 단독 누름 중 회복을 일으키는 레버의 몫이다. 전체 1,200박자 사망은 두 시작 합계 63회에서 10회로 줄었다. 여덟 몸이 하나의 통계를 공유하므로 재현 단위는 시작 2개다. 첫 시작의 기존 집계는 두 번 같았다. 두 번째 시작은 첫 시작의 선택 후 진단을 본 뒤 별도로 고정했다. 기존 9,000박자·16몸 결과는 이 표의 대조군으로 쓰지 않았다.

첫 시도에서 선택이 끝난 **뒤** 동작을 바꾼 진단도 .932 / .908, 사망 0 / 7 / 8이었다. 이 위치에서는 실제 실행과 기존 믿음의 신용 배분이 어긋날 수 있어 진단 기록으로 남긴다. 선택 전 접면도 재선택의 98.1%, 99.0%에서 기존 후보를 교체했다. 이번 결과는 이 장치에서 몸짓 이후의 속 변화가 선택에 들어갈 때의 이득을 보여 준다. 기관 사이의 적절한 중재, 한 몸의 드문 겪음으로 배우기, 숨은 원인을 분리하는 능력은 별도 판정이 필요하다. [선택 후 진단 원자료](measurements/motor_proposal.json)

## 기억과 접면

몸이 내놓는 합법적인 운동 조합의 성분을 원핫으로 풀어 성분이 시작한 박자를 기록한다. 각 성분마다 1~20박자 뒤 두 결핍의 실측 변화량을 평균한다. 통계는 박자마다 0.995배로 바랜다. 여덟 몸은 이 통계를 공유하고 각 몸의 진행 중 흔적은 분리한다. 후보 조합을 이룬 성분들의 지연 효과를 더해 현재 결핍에 적용하고, 예상 결핍 제곱 평균이 기존 후보보다 작은 조합 중 이미 실행한 것을 제안한다. 상관된 동작의 효과를 중복해서 더할 수 있다. 20박자는 설계자가 정했다. 효과가 올 정확한 시각과 실제로 어느 누름이 받아들여졌는지는 기관에 들어가지 않는다.

접면은 기존 고름이 연 재선택 박자와 허용 마스크를 받는다. 고름이 후보를 확정하고 신용을 기록하기 직전에 제안을 넣는다. 그 뒤의 원래 탐색 동작은 실제 출력을 다시 바꿀 수 있다. 학습은 최종 실행된 동작으로 한다. 새 후보가 바꾼 횟수는 중간 선택 횟수이며, 기존 출처 장부에는 이 기관 전용 출처가 없다. 결과는 별도 시험 접면에서 얻었다. [원형](research/motor_interface.py) · [고정 버전에 적용하는 선택 접면](research/motor_candidate.patch)

## 생활 경계와 호환성

상류 레버 시험기는 다시 태어날 몸의 dead 표시를 지운 뒤 그 표시로 busy를 초기화했다. 예약된 상승·하락의 시계도 남아 있었다. 이 시험기의 생애 경계에서는 먼저 마스크를 복사해 해당 몸의 세 시계를 지운다. 모든 실험군에 같은 수정이 들어간다. 인접한 살아 있는 몸의 시계 보존과 죽은 몸의 운동 흔적 절단을 단위 시험으로 확인했다. 기존 상류의 일괄 felt 초기화는 그대로여서, 다른 몸의 재탄생이 감촉 입력에 미치는 영향은 남은 점검 항목이다. Claude의 체크아웃은 수정하지 않았다.

결정론적 단일 스레드 CPU, 두 몸, 200박자에서 실행된 행동, 두 몸의 저장량, 생사, 위치의 바이트열을 매 박자 SHA-256에 넣었다. 기본·관찰만·기본 반복의 해시는 모두 `75db5645f9a1c3505d41daf146d94339f2a404488af16e55922e3cd22118c7b0`였다. 이 시험은 해당 궤적과 관측량을 검증한다. 모든 가중치, 다른 장치, 오래된 체크포인트의 호환성은 별개다. 새 기관은 이 시험에서 체크포인트에 저장하지 않는다.

후보를 실제로 켠 CPU 반복 2회도 서로 같은 해시 `262c5202f2e53ac9c92e5f822e41fc1ac69a1b20bcdcb396a82eb14fecea3b0d`를 냈다. 이 짧은 200박자에서는 원인 몫이 분리 단계 1.00, 반전 단계 .25였다(기본 .25, .50). 모든 갈래의 사망은 0이었다. 짧은 호환성 시험에서는 반전을 이기지 못했다. 죽는 경계와 살아 있는 이웃은 별도 회귀 시험에서 검사했다. [기본 궤적](measurements/motor_audit_native.json) · [관찰 전용](measurements/motor_audit_shadow.json) · [개입 반복](measurements/motor_audit_active_repeat.json) · [생애 경계 패치](research/lever_timer_cut.patch)

기본 레버 몸의 세 운동축은 6×5×3이다. 8몸에서 이 기관이 유지하는 텐서들의 합은 **18,514바이트**다. 중간 텐서, 프레임워크, ATANOR 본체와 체크포인트는 이 수치 밖이다. 입력 접면은 현재 방의 두 바탕 채널을 Going과 같은 순서로 읽는다. 새 몸이 이 두 채널을 다른 뜻으로 재배선했을 때의 정렬은 검증하지 않았다.

## 재현 조건

ATANOR [ee563d7](https://github.com/Cozystone/ATANOR_The-Creator/tree/ee563d7b0289613bf6b53c6bd2a097aeddd22d72), torch 2.11.0+cu128, 데스크톱 RTX 5080. 일반 시험은 8몸, 1,200박자, 단계 경계 240/720, eye 1.1, 상류의 기본 환경 깃발. 옛 보고의 16몸·9,000박자와 A2 주선의 F2 설정은 별도 조건이다. 입출력 비교용 CPU 시험은 2몸, 200박자, 경계 40/120이다. 여러 프로세스가 같은 기기를 사용했으므로 JSON의 경과 시간으로 성능·전력을 비교할 수 없다.

실행에는 ATANOR 작업 디렉터리의 s3_e/mind.pt, s3_e/going.pt, mc/mc_depth.pt와 ATANOR_DEPTH_PATH의 selfsup.py, model.py가 필요하다. 공개 핀에서 빠진 전이 의존성 tools/a1_data.py는 기존 연구 체크아웃의 사본을 사용했다. 현장 파일들의 SHA-256은 아래와 같다.

| 파일 | SHA-256 |
|---|---|
| mind.pt | 4e626ef8b40922415c5ea1e91acc9d5175b5165a7601fb08d64c669da85555de |
| going.pt | 22357edeade7221a29c60902fe7d3109f07e72a79aac734944857f56d91ade33 |
| mc_depth.pt | ffa0e3cbbbfdcd25271ae0baaef563ef43fb08aec031d842528df76be1b49b0e |
| tools/a1_data.py | b0cae7b0d0eda42a3ed8e8b2c9407f07cbadac527427c824156894988ee8a997 |

```powershell
# From the checkpoint working directory. Use a disposable checkout at the pin.
git -C $Atanor apply --check "$Loki/research/motor_candidate.patch"
git -C $Atanor apply "$Loki/research/motor_candidate.patch"
$env:PYTHONPATH = $Loki
python "$Loki/research/motor_interface.py" $Atanor s3_e native.json --native --seed 9781
python "$Loki/research/motor_interface.py" $Atanor s3_e candidate.json --at-choice --seed 9781
```

Use a new process per run. Omit the mode for a read-only observer. Add --audit for the CPU trajectory check. The earlier --active diagnostic intercepts after the chooser has already recorded credit. Its numbers describe a different intervention. Only --at-choice is the candidate for integration.

## Claude와 맞출 다음 판정

레버의 행동 단위 접면은 주머니 방 b에도 시험할 수 있지만 둘의 원인이 같다는 증거는 아직 없다. 맞는 위치를 말한 46몸 중 그 위치를 후보로 든 몸은 23이었다. 반대로 첫 채움 31건 중 23건에는 사전에 해당 후보가 없었다. 음식 옆까지 갔다 죽은 13몸은 가까이 서기, 정면 맞추기, 손 시도 중 어느 경계에서 끊겼는지 따로 봐야 한다. 상류의 정면 판정은 걸음 뒤에 읽어 12몸만 잡았다. 우선 같은 상태에서 한 걸음 전·접촉을 판정한 순간·뒤의 방향과 거리, 실제 실행된 손, 속 변화, 죽음 마스크를 연결하자. 그다음 동일한 상태를 복제해 방향만 맞춰 준 경우, 손 시도만 준 경우, 둘 다 준 경우, 원래 선택을 비교하면 눈과 손 중 어디서 끊기는지 가를 수 있다. 정답은 판정용 사본에서만 사용한다. [ATANOR의 첫 만남 기록](https://github.com/Cozystone/ATANOR_The-Creator/blob/ee563d7b0289613bf6b53c6bd2a097aeddd22d72/tools/results/pk3b_chain2_bundle.json)

A2에서 닫은 강제 장소 재배치·여정 단위 감·눈이 걷기를 이끄는 묶음은 기본 꺼짐을 유지한다. Q1은 대기가 줄어든 후보로 둔다. RL1은 깨어 있는 지연을 개선했으나 쉬는 계산과 첫 배선 학습 손실을 함께 판정해야 한다. 어떤 기관을 주선에 싣기 전에는 여러 독립 시작의 긴 레버 생애, A2의 사망·채움, 결과와 행동의 시간 짝을 섞은 대조까지 필요하다. 현재 측정은 단일 몸 적응, 배터리 소모, 안경 보행, 장기 기억 보존을 다루지 않는다.
