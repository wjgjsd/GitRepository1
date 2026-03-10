# [Project Report] 맥락 주입형 실시간 AI 자막 생성 시스템 (CARSS)
### : Adaptive ASR with Semantic ControlNet & RAG-Driven Dynamic Refinement

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 기존 범용 음성 인식(ASR) 모델이 해결하지 못하는 **전문 도메인(수학, 공학, 의학 등)의 고유 명사 오인식** 및 **비원어민의 특이 억양(인도, 한국식 발음 등)** 문제를 해결하기 위한 '지능형 자막 생성 시스템'이다. 2024년 발표된 최신 SOTA 모델인 **NVIDIA Canary-1B**를 핵심 엔진으로 채택하고, 사용자가 입력한 **'맥락(Context)'** 정보를 모델의 디코딩 과정에 직접적으로 개입시켜, 오디오 신호가 불분명한 상황에서도 정확한 단어를 선택하도록 유도하는 것을 목표로 한다.

## 2. 핵심 컨셉 (Core Concept)
* **ControlNet-inspired Conditioning:** 이미지 생성 AI의 ControlNet처럼, 고정된 모델 옆에 맥락을 전담하는 어댑터를 두어 출력 결과를 제어한다.
* **No-Audio Knowledge Distillation:** 실제 음성 녹음 데이터 없이, 텍스트 기반의 전문 지식을 '가상 음성 합성(Synthetic Audio)'을 통해 ASR 모델에 이식한다.
* **Knowledge-Augmented Correction:** LLM을 새로 학습시키는 대신, RAG(검색 증강)를 통해 실시간으로 전문 사전을 참조하여 오답을 교정한다.

## 3. 시스템 구조 (System Architecture)
본 시스템은 사용자 기기(Client)와 추론 서버(Server)로 나뉘며, 4070 Ti(12GB VRAM) 환경에서 최적의 퍼포먼스를 내도록 설계되었다.

### 3.1. 클라이언트 (Client Side)
* **기술 스택:** Python, PySide6(UI), PyAudio
* **기능:** - **오디오 캡처:** 시스템 루프백 캡처를 통해 영상의 소리를 30ms 단위 청크로 분할.
  - **맥락 인터페이스:** 사용자가 현재 영상의 주제(예: "머신러닝"), 화자(예: "인도인"), 키워드 리스트를 입력.
* **통신:** WebSocket을 통한 실시간 오디오 및 메타데이터 스트리밍.

### 3.2. 서버 (Server Side)
* **기술 스택:** FastAPI, NVIDIA Triton Inference Server, Faiss (Vector DB)
* **기능:** - **ASR Engine:** Canary-1B + Semantic Adapter 가동.
  - **Knowledge Base:** 도메인별 전문 용어 임베딩이 저장된 고속 벡터 검색 엔진 운영.
  - **Refiner:** 경량화된 LLM(Gemma-2B)을 이용한 최종 자막 교정.

## 4. AI 파이프라인 (AI Pipeline: End-to-End)



1. **VAD (Voice Activity Detection):** Silero VAD를 통해 유효 목소리 구간만 필터링하여 서버 연산 낭비 방지.
2. **Context-Aware ASR (Inference):** - 사용자의 맥락 텍스트가 **Semantic Adapter**를 통해 벡터화되어 Canary-1B 디코더의 Cross-Attention 레이어에 주입됨.
   - 오디오 신호가 "Derive"인지 "The live"인지 모호할 때, 맥락(수학) 가중치에 의해 "Derive" 토큰의 확률(Logit)을 강제로 높임.
3. **RAG Retrieval:** ASR이 출력한 단어 중 신뢰도가 낮은 구간을 전문 용어 사전에서 검색하여 유사 발음의 올바른 용어 후보를 추출.
4. **LLM Refinement (In-Context Learning):** - LLM이 [ASR 초안 + 사전 검색 단어 + 맥락 지침]을 입력받아 문맥상 가장 자연스러운 문장으로 재구성.
5. **Output Delivery:** 0.5초 미만의 Latency로 클라이언트에 최종 자막 패킷 전송.

## 5. 학습 및 최적화 전략 (Training & Optimization Strategy)

### 5.1. 가상 음성 기반 지식 전이 (Synthetic Data Strategy)
실제 음성 데이터 없이 모델을 학습시키기 위한 고도화된 전략이다.
* **TTS 합성:** 도메인 텍스트를 다중 화자 TTS로 변환하여 텍스트-음성 쌍을 무한 생성.
* **정보 파괴 학습(Noise Augmentation):** 생성된 음성에 강한 배경 소음, 잔향(RIR), 주파수 차단을 적용. 모델이 소리만 들어서는 정답을 맞힐 수 없게 만들어, **함께 제공되는 '맥락 프롬프트'에 의존하도록 강제로 훈련**시킴.

### 5.2. 하드웨어 최적화 (For RTX 4070 Ti 12GB)
* **LoRA (Low-Rank Adaptation):** 전체 파라미터가 아닌 1% 미만의 어댑터 레이어만 학습하여 VRAM 점유 최소화.
* **Model Quantization:** Canary-1B(ASR)와 Gemma-2B(LLM)를 모두 4-bit/8-bit로 양자화하여 단일 GPU 메모리 내에 동시 적재.
* **8-bit Optimizer:** 학습 시 가중치 업데이트에 필요한 메모리를 절반으로 줄여 12GB 내에서 안정적 학습 보장.

## 6. 예상되는 챌린지 (Key Challenges)
* **실시간 지연 시간(Latency):** ASR-RAG-LLM으로 이어지는 다단계 구조의 지연 시간.
  - *해결:* ASR 초안을 먼저 출력하고 LLM 교정본을 나중에 덮어쓰는 **Double-Buffering** 방식 채택.
* **LLM 환각(Hallucination):** 맥락에 심취해 화자가 하지 않은 말을 지어낼 위험.
  - *해결:* 교정 전후 단어의 발음 유사도(Phonetic Distance)를 체크하여 일정 범위를 벗어나는 수정은 기각하는 제약 로직 적용.
* **맥락 주입의 과적합:** 맥락 힌트가 너무 강해 오디오 신호를 무시하는 현상.
  - *해결:* 추론 시 맥락 벡터의 영향력을 조절할 수 있는 **Scaling Factor** 파라미터 도입.

## 7. 기대 효과 및 차별성 (Expected Outcomes & Technical Edge)
* **압도적인 하드웨어 효율성:** 4비트/8비트 양자화 기술을 적용하여 **RTX 4070 Ti(12GB VRAM) 한 대만으로도 전체 파이프라인(VAD+ASR+LLM+Embedding) 구동이 가능**함. (전체 VRAM의 약 60~70% 점유로 실시간 추론 안정성 확보)
* **데이터 주권 확보:** 고가의 음성 녹음 데이터 없이 '텍스트 용어집'만으로 특정 분야의 전문가 AI를 즉시 구축 가능.
* **결정론적 모델 제어:** 단순 확률에 의존하는 기존 방식과 달리, ControlNet 스타일의 어댑터를 통해 개발자가 맥락 반영 강도를 직접 제어할 수 있음.
* **현장 적용성:** 저사양 사용자 환경에서도 서버의 고성능 GPU 자원을 활용해 전문 강의 및 세미나 자막을 실시간 제공 가능.

## 8. 사용 모델 명세 및 역할 (Model Specifications)

| 분류 | 모델명 | 출시/업데이트 | 주요 역할 |
| :--- | :--- | :--- | :--- |
| **VAD** | Silero VAD v4 | 2023. 11. | 무음 제거를 통한 연산 자원 최적화 |
| **ASR** | NVIDIA Canary-1B | 2024. 02. | 맥락 주입 어댑터가 부착된 메인 인식 엔진 |
| **LLM** | Google Gemma-2B | 2024. 02. | 4-bit 양자화 기반 지능형 문장 교정 및 정제 |
| **Embedding**| BGE-M3 | 2024. 01. | RAG 지식 검색을 위한 고정밀 벡터 추출 |
| **TTS** | Coqui TTS (VITS) | 2023. 08. | 가상 학습 데이터 생성을 위한 음성 합성 엔진 |
