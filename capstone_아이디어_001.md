# [Final Project] 맥락 주입형 실시간 AI 자막 생성 시스템 (CARSS)
## : Adaptive ASR with Semantic Conditioning & Dynamic Correction

## 1. 개요 (Overview)
본 프로젝트는 2024년 발표된 SOTA 모델인 **NVIDIA Canary-1B**를 기반으로 한다. 단순히 소리를 텍스트로 바꾸는 기존 방식을 넘어, **ControlNet 아키텍처에서 영감을 얻은 추가 컨디셔닝 레이어**를 통해 사용자 맥락(주제, 전문 용어)을 음성 인식 과정에 강제로 주입하고, 실시간으로 인식 오류를 교정하는 고성능 자막 시스템 구축을 목표로 한다.

---

## 2. 핵심 기술 아키텍처 (Core Technical Architecture)

### 2.1. ASR: Semantic ControlNet Adapter (핵심 차별점)
기본 ASR 모델의 가중치를 고정(Freeze)한 채, 이미지 생성 모델의 ControlNet처럼 **맥락 정보를 처리하는 별도의 어댑터 구조**를 설계한다.
* **구조:** Canary-1B의 Encoder와 Decoder 사이에 사용자의 '맥락 텍스트'를 인코딩한 벡터를 주입하는 **Cross-Attention Adapter** 추가.
* **작동:** 사용자가 "미적분" 입력을 주면, 어댑터가 "Differential", "Integral" 등 관련 토큰의 가중치를 물리적으로 증폭시켜 오디오가 모호해도 맥락에 맞는 단어가 우선 출력되도록 제어.
* **장점:** 전체 모델 학습 없이 작은 어댑터 레이어만 학습하므로 4070 Ti(12GB)에서 충분히 학습 및 실시간 추론 가능.

### 2.2. Post-Editing: RAG 기반 Dynamic Correction (LLM 학습 대체안)
4070 Ti에서 8B 이상급 LLM을 미세 조정(Fine-tuning)하는 부담을 줄이기 위해, **학습 대신 '검색 증강(RAG)'과 'In-Context Learning'**을 결합한다.
* **방법:** 1. 사용자가 선택한 도메인(예: 수학)의 **용어집(Glossary)**을 벡터 DB에 실시간 로드.
  2. ASR 결과물이 나오면, LLM(Gemma-2B 또는 4-bit 양자화된 Llama-3)에게 **[ASR 결과 + 검색된 올바른 용어 + 사용자 맥락]**을 프롬프트로 전달.
  3. LLM은 추가 학습 없이도 프롬프트에 담긴 정보를 바탕으로 'The live'를 'Derive'로 즉시 교정.
* **장점:** 학습 서버 없이도 도메인 지식을 무한히 확장 가능하며 VRAM 점유율 최소화.

---

## 3. AI 파이프라인 (Detailed Pipeline)

1. **Audio Stream Capture:** 시스템 루프백 오디오 캡처 (Python PyAudio).
2. **VAD (Voice Activity Detection):** Silero VAD를 이용한 발화 구간 분리.
3. **Context Injection (ControlNet Layer):** 사용자 입력 맥락을 임베딩하여 ASR Decoder의 Attention 레이어에 바이어스(Bias)로 주입.
4. **Contextual ASR Inference:** Canary-1B + Semantic Adapter를 통한 초안 자막 생성.
5. **Dynamic Knowledge Retrieval:** 초안 자막 내 모호한 단어를 도메인 사전에서 검색하여 후보군 추출.
6. **Prompt-based Refinement:** 경량 LLM이 프롬프트 지시사항에 따라 최종 자막 확정.

---

## 4. 실현 가능성 및 하드웨어 최적화 (4070 Ti 12GB 기준)

* **VRAM 배분 전략 (Total 12GB):**
    - **Canary-1B (ASR):** FP16 기준 약 2~3GB 점유.
    - **Gemma-2B (Refiner):** 4-bit 양자화 적용 시 약 1.5~2GB 점유.
    - **ControlNet Adapter & Vector DB:** 약 1GB 미만 점유.
    - **여유 공간:** 시스템 및 실시간 스트리밍 버퍼용으로 약 6~7GB 확보 가능.
* **학습 전략:**
    - ASR 어댑터는 **PEFT(LoRA)** 방식을 사용하며, 가상으로 생성된 `[오디오 특징 - 맥락 텍스트 - 정답]` 데이터를 통해 4070 Ti에서 수 시간 내 학습 완료 가능.

---

## 5. 핵심 챌린지 및 대응 방안

### 5.1. 실시간성 (Latency) 관리
- **문제:** ASR 이후 LLM 교정 단계가 추가되어 자막 지연 발생 우려.
- **대응:** LLM 교정을 매 단어마다 하지 않고, 문장 마디(Chunk) 단위로 비동기 병렬 처리. ASR 초안은 즉시 출력하고 LLM 교정본은 0.5초 내에 업데이트하는 '이중 업데이트' 방식 채택.

### 5.2. 컨디셔닝 레이어의 학습 데이터 부재
- **문제:** 오디오 신호와 맥락 텍스트 간의 연관성을 학습시킬 데이터 부족.
- **대응:** TTS(Text-to-Speech)를 활용하여 특정 도메인(수학 등)의 인공 음성을 대량 생성하고, 여기에 맥락 태그를 붙여 어댑터를 학습시키는 **가상 데이터 파이프라인** 구축.

---

## 6. 결론 및 기대 성과
본 프로젝트는 **최신 SOTA 모델(Canary-1B)**, **ControlNet 스타일의 가이드 주입**, 그리고 **학습 효율을 극대화한 RAG 기반 교정**을 결합한 혁신적인 시도이다. 이를 통해 전문적인 환경(대학 강의, 기술 세미나)에서 사용자 맞춤형 초고정밀 자막 서비스를 실시간으로 제공할 수 있다.
