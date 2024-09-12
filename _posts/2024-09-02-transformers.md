---
title:  "🤗 Transformers 시작하기: pipeline으로 사전 훈련된 모델 추론하기"
categories: 
  - Hugging-Face
tag:
  - transformers
  - pipeline
---


## 🤗 Transformers란?
- PyTorch, TensorFlow, JAX를 위한 최첨단 머신러닝
- **사전학습된 최첨단 모델**들을 쉽게 다운로드하고 훈련시킬 수 있는 API와 도구를 제공함
- 다양한 분야의 task를 지원함
    - 📝 자연어 처리: 텍스트 분류, 개체명 인식, 질의응답, 언어 모델링, 요약, 번역, 객관식 질의응답, 텍스트 생성
    - 🖼️ 컴퓨터 비전: 이미지 분류, 객체 탐지, 객체 분할
    - 🗣️ 오디오: 자동음성인식, 오디오 분류
    - 🐙 멀티모달: 표 질의응답, 광학 문자 인식 (OCR), 스캔한 문서에서 정보 추출, 비디오 분류, 시각 질의응답
- PyTorch, TensorFlow와 JAX 간의 상호운용성을 지원하여, 모델의 각 단계마다 유연하게 다른 프레임워크 사용 가능함   
-> ex) 코드 3줄만 써서 모델을 훈련시킨 다음, 다른 프레임워크 상에서 추론


## pipeline이란?
- 사전 학습된 모델과 토크나이저를 자동으로 다운로드하고 캐시하여, 쉽고 빠르게 **추론**할 수 있게 하는 도구
- `pipeline()`: hugging face의 transformers 라이브러리에서 제공하는 함수로, **사전 학습된 모델을 쉽게 불러와** 다양한 NLP 작업을 수행할 수 있도록 도와준다.
- 여러 task에서 pipeline()을 즉시 사용할 수 있음
- 지원하는 task 예시

| **태스크**     | **설명**                                                            | **모달리티**     | **파이프라인 ID**                             |
|----------------|---------------------------------------------------------------------|------------------|-----------------------------------------------|
| 시각 질의응답  | 주어진 이미지와 이미지에 대한 질문에 따라 올바르게 대답하기         | 멀티모달         | pipeline(task="vqa")                          |
| 문서 질의응답  | 주어진 문서와 질문에 대해 올바르게 대답하기         | 멀티모달         | pipeline(task="document-question-answering")                          |
| 이미지 캡션 달기  | 주어진 이미지의 캡션 생성하기         | 멀티모달         | pipeline(task=“image-to-text”)                         |

- [Transformers documentation: Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)에서 다양한 task의 pipeline() 정보 및 예시들을 볼 수 있다.


## 용어 정리
- **토크나이저(Tokenizers)**: NLP에서 텍스트 데이터를 모델에 입력할 수 있는 형태로 변환하는 도구 (즉, 텍스트를 숫자 형태의 벡터로 변환해주는 도구)
    - hugging face의 `transformers` 라이브러리에서 `pipeline()`을 호출할 때, 토크나이저는 모델과 함께 자동으로 설정되며, 입력된 텍스트를 모델이 이해할 수 있는 형태로 변환하는 역할을 함
    - ex) 감정 분석 task: 입력된 텍스트는 먼저 토크나이저를 통해 토큰화된 후 모델에게 전달됨
- **캐시하다**: 다운로드된 모델과 토크나이저를 로컬에 저장하여, 나중에 다시 사용할 때 재다운로드 없이 빠르게 접근할 수 있도록 한다.
    - 본 의미: 특정 데이터를 임시로 저장해두어 나중에 빠르게 접근할 수 있도록 하는 것
    - `pipeline()`이 사전 훈련된 모델과 토크나이저를 다운로드하면 로컬 시스템의 특정 디렉토리에 저장되는데, 이 저장된 데이터가 **캐시**이다.
    - 이후 동일한 모델이나 토크나이저를 다시 사용할 때, 처음부터 다시 다운로드할 필요 없이 이미 캐시된 데이터를 불러오기 때문에 더 빠르게 작업 수행할 수 있다.


## 🤗 Transformers 시작하기: pipeline으로 사전 훈련된 모델 추론하기
- pipeline을 사용하여 추론하고, 사전학습된 모델과 전처리기를 AutoClass로 로드하고, PyTorch 또는 TensorFlow로 모델을 빠르게 학습시키는 방법을 알아보겠다.
- [Transformers documentation: 둘러보기](https://huggingface.co/docs/transformers/main/ko/quicktour)를 따라 진행했으며, 코드 실행은 해당 페이지에서 제공하는 colab notebook을 다운로드해서 colab 환경(CPU)에서 진행했다.
- [Transformers documentation: Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)에서 'DocumentQuestionAnsweringPipeline' 부분의 pipeline으로 진행했으며, [impira/layoutlm-document-qa](https://huggingface.co/impira/layoutlm-document-qa#getting-started-with-the-model) 모델을 사용했다. (이 두 링크에 제시된 코드를 기반으로 진행함)

#### 1. 필요한 라이브러리 설치 및 확인하기
- hugging face의 해당 모델 페이지(ex: [impira/layoutlm-document-qa](https://huggingface.co/impira/layoutlm-document-qa#getting-started-with-the-model))에 들어가면 아래와 같이 모델 사용 전에 미리 설치해야 할 라이브러리들이 언급되어있다.
![]({{site.url}}/images/2024-09-02-transformers/start.png)
- 모델 사용하기 전에 다음 명령어를 통해 요구 라이브러리들을 설치하면 된다.
```python
!pip install Pillow pytesseract torch transformers
```
- 이때 선호하는 머신러닝 프레임워크도 설치해야 한다.
```python
!pip install torch
```

#### 2. pipeline의 instance 생성하기
- document question answering task와 impira/layoutlm-document-qa 모델을 사용한  pipeline을 생성했다.
```python
from transformers import pipeline
document_qa = pipeline(model="impira/layoutlm-document-qa")
```
- **pipeline 인스턴스 생성 방법**
    - **task 유형 지정(기본 사용)**: task 유형(ex: "document-question-answering")만 지정하면, 해당 task에 맞는 기본 모델이 자동으로 사용된다.
    ```python
    from transformers import pipeline
    document_qa = pipeline("document-question-answering")
    ```
    - **특정 모델 지정**: 모델 이름을 명시하여 특정 모델을 사용할 수 있다.
    ```python
    from transformers import pipeline
    document_qa = pipeline(model="impira/layoutlm-document-qa")
    ```

    - **작업 유형과 모델 모두 지정**
    ```python
    from transformers import pipeline
    document_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
    ```

#### 3. pipeline 인스턴스로 추론하기
- 아래 코드를 실행하여 LayoutLM 모델을 사용한 문서 이미지 기반 질의 응답 작업을 수행할 수 있다.
```python
document_qa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice number?"
)
```
- **problem 1.**: **Tesseract-OCR 엔진**이 시스템에 설치되어 있지 않아서 발생한 오류이다.
![]({{site.url}}/images/2024-09-02-transformers/error1.png)
- **solution**: **Tesseract-OCR**을 설치하고, Tesseract-OCR을 파이썬에서 사용하기 위해 **pytesseract** 라이브러리도 설치한다. 그 다음, **런타임 > 세션 다시 시작**을 선택하여 python 환경을 재시작하면 된다.
```python
!apt-get install -y tesseract-ocr
```
```python
!pip install pytesseract
```

Colab 환경에서 오류 발생했을 때, 해결하기 위해 라이브러리를 설치해도 문제가 그대로라면 **런타임을 다시 시작**해보자!
{: .notice--success}

- **추론 결과**: [Transformers documentation: Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)에서의 example 결과랑 동일함을 확인할 수 있다.
![]({{site.url}}/images/2024-09-02-transformers/sol.png)


## Reference
- [시작하기: 🤗 Transformers](https://huggingface.co/docs/transformers/main/ko/index)
    - 🤗 Transformers에서 지원하는 모델 확인
- [시작하기: 둘러보기](https://huggingface.co/docs/transformers/main/ko/quicktour)
    - 🤗 Transformers 시작하기
    - 사전학습된 모델 로드 및 학습
- [Transformers documentation: Pipelines](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)
    - pipeline() 관련 지원하는 태스크의 전체 목록 확인