---
title:  "Computer Vision Using Convolutional Neural Networks"
categories: 
  - Deep-Learning
tag:
  - CV
  - CNN
---

## Introduction
CNN을 사용한 computer vision의 전반적인 내용을 간단히 다룰 것이다. CNN의 개념과 주요한 CNN 구조들, 그리고 computer vision task들에 대해 설명하겠다. CV 분야에서도 transformer가 많이 사용되고 있지만, CNN에 대한 지식을 갖고 있는 것은 필수적이기 때문에 알고 넘어가자!  
\<Hands-On Machine Learning\>의 ch14를 기반으로 중요하다고 생각되는 개념을 중점으로 기재했으니, 이 외의 내용은 해당 책의 내용을 참고하면 된다.

## 1. 시각 피질(visual cortex) 구조
![]({{site.url}}/images/2024-11-12-CV using CNN/receptive.png)
- 시각 피질 안의 많은 뉴런이 **receptive field(수용장)**를 가짐
- **receptive field**: 각 뉴런이 반응하는 시각적 입력의 영역으로, 뉴런들이 시야의 일부 범위 안에 있는 시각 자극에만 반응함.
- **계층적 구조**: 시각 피질은 여러 계층으로 이루어져 있음. 하위 계층의 뉴런들은 작은 영역에서 line, edge 같이 단순한 패턴을 감지하고, 상위 계층으로 갈수록 점차 복잡한 형태, 색상, 움직임 등을 감지하는 뉴런들이 활성화됨.
- **수용장의 확장**: 하위 계층의 뉴런들은 작은 영역에 대한 정보를 처리하는 반면, 상위 계층의 뉴런들은 더 넓은 수용장을 통해 정보를 통합함. 즉, 상위 계층으로 갈수록 수용장이 확장되는 것임.  
-> why? 여러 하위 뉴런들의 출력을 통합하여 넓은 범위의 정보를 결합하기 때문
- 고수준 뉴런은 이웃한 저수준 뉴런의 출력에 기반함 *(수용장의 확장 이유)*
- CNN은 이러한 시각 피질 구조를 기반으로 영감을 받아 설계됨. CNN의 주요 구성요소는 **convolution layer**와 **pooling layer**임.

## 2. convolution layer(합성곱 층)
- **1st conv layer의 뉴런**: 합성곱 층 뉴런의 수용장 안에 있는 픽셀에만 연결됨 (not 입력 이미지의 모든 픽셀)
- **2nd conv layer의 뉴런**: 1st 층의 출력(feature map)에서 해당하는 수용장에만 연결됨. (not 전체 입력) 즉, 1st 층의 local 패턴들에 대한 정보를 수용장 내에서 통합해 더 높은 수준의 feature를 학습함.
- **계층적 구조**: 하위 계층에서는 작은 저수준 feature에 집중하고, 상위 계층으로 갈수록 더 큰 고수준 feature에 집중함.

#### 2.1  filter
- 뉴런의 가중치: 수용장 크기의 작은 이미지로 표현될 수 있음. (여기서 뉴런은 픽셀 하나에 해당함)  
*그냥 filter = 가중치 = receptive field 라고 이해하자!* 
- 층의 전체 뉴런에 적용된 **하나의 filter는 하나의 feature map을 만듦** (CNN은 가중치를 공유한다!)

#### 2.2 여러 가지 feature map 쌓기
- 실제 합성곱 층은 여러 가지 filter를 갖고 filter마다 하나의 feature map을 출력함 -> 3D로 표현해야 함!  
(*책에서의 이러한 표현은 오해의 여지가 있음. **filter의 개수 = feature map의 channel수**임. 즉, feature map은 1개이고, feature map의 channel수가 filter의 개수에 의해 결정되는 것임. 그러므로 소주제도 '여러가지 feature map 쌓기'보다는 'feature map의 channel(depth) 쌓기'가 나을 것 같음*)
- 하나의 뉴런: 각 feature map의 pixel
- 하나의 feature map(*하나의 feature map 안의 하나의 channel*) 안에서는 모든 뉴런이 같은 파라미터(동일한 가중치와 편향)를 공유함 (feature map의 다른 channel에 있는 뉴런은 다른 파라미터를 사용함)
- **정리**: 하나의 합성곱 층이 입력에 여러 filter를 동시에 적용하여 입력에 있는 여러 feature를 감지할 수 있음