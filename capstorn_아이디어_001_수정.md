# [Final Proposal] 맥락 주입형 실시간 AI 자막 생성 시스템 (CARSS)
### : Semantic ControlNet ASR & Knowledge-Augmented Dynamic Refinement

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 NVIDIA Canary-1B를 기반으로, 사용자의 **맥락(Context)** 정보를 시스템 전 과정(음성 인식, 지식 검색, 문장 교정)에 결합하는 **지식 중심(Knowledge-centric) 실시간 자막 시스템**이다. 전문 용어 오인식 문제를 해결하기 위해 '물리적 어댑터'와 '가상 데이터 학습' 기술을 통합한다.

## 2. 핵심 컨셉 (Core Concept)
* **ControlNet-inspired Conditioning:** 고정된 ASR 모델 옆에 맥락 전담 어댑터를 두어 출력 결과를 제어.
* **No-Audio Knowledge Distillation:** 음성 데이터 없이 텍스트 지식을 TTS와 오디오 파괴 기법으로 모델에 이식.
* **Knowledge-Augmented Correction:** RAG를 통해 실시간 전문 사전을 참조하여 LLM이 오답을 교정.

## 3. 시스템 구조 (System Architecture)
* **Client:** Python 기반 오디오 캡처 및 맥락 입력 인터페이스, 자막 오버레이 UI.
* **Server:** FastAPI 및 NVIDIA Triton을 활용한 ASR/LLM 통합 추론 엔진 및 Faiss 벡터 DB.

## 4. AI 파이프라인 (AI Pipeline)
1. **VAD:** 실시간 발화 구간 추출.
2. **Context-Aware ASR:** Canary-1B + Semantic Adapter를 통한 맥락 편향 인식.
3. **RAG Retrieval:** 신뢰도 낮은 단어를 벡터 DB에서 검색하여 전문 용어 후보 매칭.
4. **LLM Refinement:** 프롬프트 기반 문맥 교정 및 최종 자막 확정.

## 5. 학습 및 최적화 전략 (Training & Optimization)
* **Synthetic Audio Pipeline:** 도메인 텍스트를 TTS로 합성 후 노이즈를 주입하여 맥락 의존도 학습 데이터 생성.
* **Context-Priority LoRA:** 디코더 Attention 레이어에 LoRA를 적용하여 맥락 힌트에 높은 가중치를 두도록 훈련.
* **8-bit Optimizer:** 학습 시 메모리 사용량을 절반으로 줄여 소비자용 GPU에서도 대형 모델 학습 가능하게 구현.

## 6. 예상되는 챌린지 (Key Challenges)
* **Latency 제어:** 비동기 처리 및 Double-Buffering UI를 통한 체감 지연 시간 최소화.
* **Hallucination 방지:** 음성학적 유사 거리(Phonetic Distance) 제약을 통한 무분별한 수정 방지.

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
