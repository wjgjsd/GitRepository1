# [Project] 맥락 주입형 실시간 AI 자막 생성 시스템 (CARSS)

## 1. 개요 (Overview)
본 프로젝트는 전문 용어 오인식 및 화자 억양 문제를 해결하기 위해, 영상의 주제와 특징을 '맥락(Context)'으로 입력받아 추론에 개입시키는 지식 중심(Knowledge-centric) 실시간 자막 시스템이다.

## 2. 핵심 컨셉 (Core Concept)
* **No-Audio Training:** 실제 음성 녹음 데이터 추가 없이, 텍스트 기반의 도메인 지식과 맥락 힌트만으로 인식률을 교정한다.
* **Contextual Prioritization:** 오디오 신호가 모호할 때, 사용자가 입력한 맥락(예: 수학, 인도인 억양 등)에 더 높은 가중치를 두어 단어를 선택한다.

---

## 3. 시스템 구조 (System Architecture)

### 3.1. 클라이언트 (Client Side)
* **기술 스택:** Python (PySide6 / Flet), PyAudio, WebSockets
* **기능:** 시스템 오디오 루프백 캡처, 실시간 스트리밍 전송, 자막 오버레이 UI.
* **사용자 입력:** 영상 특징(주제: 미적분, 화자: 인도인 억양, 전문 키워드 등).

### 3.2. 서버 (Server Side)
* **기술 스택:** FastAPI (Asynchronous), NVIDIA Triton Inference Server, Docker
* **기능:** 무거운 AI 추론 수행 및 사용자별 맞춤형 **LoRA(Adapter) 모듈** 실시간 로드.
* **확장성:** 도메인이 늘어나도 전체 모델 교체 없이 작은 어댑터 파일만 추가하여 확장 가능.

---

## 4. AI 파이프라인 (AI Pipeline)



1. **오디오 전처리 (VAD):** Silero VAD를 통한 목소리 구간 추출 및 서버 연산 효율화.
2. **맥락 주입 음성 인식 (Context-Biased ASR):** - 모델: NVIDIA Canary-1B 또는 Faster-Whisper.
   - 원리: 사용자 입력 텍스트를 프롬프트(Prompt)로 주입하여 도메인 특화 용어의 발생 확률(Logits)을 강제로 상향 조정.
3. **LLM 맥락 정제 (Contextual Post-Editing):**
   - 모델: Gemma-2B / Llama-3-8B (경량 LLM).
   - 원리: ASR의 초안 결과물에서 맥락과 일치하지 않는 오인식 단어를 사용자 힌트를 바탕으로 최종 교정. (예: 'The live' → 'Derive')
4. **도메인 특화 번역 (Domain-Specific Translation):**
   - 정제된 텍스트를 맥락 기반 전문 용어 사전을 참조하여 정확한 타겟 언어로 변환.

---

## 5. 학습 및 모듈 확장 전략 (Training Strategy)

**핵심: 소리 데이터가 아닌 '맥락 이해 및 반영 능력'을 학습시키는 것에 집중함.**

### 5.1. 프롬프트 순응 학습 (Instruction Tuning)
* **방법:** ASR 모델의 Decoder 부분에 LoRA(Low-Rank Adaptation) 적용.
* **목표:** 모델이 입력받은 '맥락 텍스트'를 배경지식으로 강력하게 인지하여, 불분명한 발음을 맥락에 맞는 단어로 매핑하는 의존도를 높임.

### 5.2. 가상 오인식 교정 학습 (Synthetic Error Correction)
* **방법:** 텍스트 기반 데이터 합성 및 LLM 미세 조정.
* **프로세스:**
  1. 전문 분야(수학, 의학 등) 텍스트 코퍼스 수집.
  2. 음성 인식에서 흔히 발생하는 발음 유사 오타(Phonetic Error) 생성.
  3. `[오타 문장] + [맥락 키워드]`를 입력 시 `[정답 문장]`을 도출하도록 LLM 학습.

---

## 6. 기대 효과 및 차별성
* **실시간 전문성:** 사용자가 "이 영상은 인도 수학 강의다"라고 알려주는 순간, AI는 해당 도메인의 전문가 모드로 즉시 전환됨.
* **데이터 효율성:** 대규모 음성 데이터 구축 없이 관련 텍스트와 용어집만으로 전용 어댑터(LoRA) 생성 가능.
* **서버 기반 서비스:** 저사양 사용자 기기에서도 서버의 고성능 GPU 자원을 활용한 고품질 자막 이용 가능.
