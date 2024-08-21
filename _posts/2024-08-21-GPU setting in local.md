---
title: "로컬에서 딥러닝을 위한 GPU 개발환경 구축하기"
categories: 
  - Deep-Learning
tag:
  - GPU
  - CUDA
---


## Introduction
로컬 환경에서 vscode를 사용해 딥러닝 모델을 실행하려고 할 때, GPU 설정이 필요하다. Colab과 같이 GPU가 있는 서버 환경에서는 GPU 설정이 자동으로 이루어져 복잡한 과정 없이 바로 사용할 수 있지만, 로컬에서는 직접 설정을 해줘야 한다. 과정이 꽤 복잡하기 때문에 로컬 환경에서 GPU를 사용하기 위한 방법을 정리하고자 한다.


## Goal
vscode의 terminal(cmd)에서 위 코드를 실행했을 때, 결과가 `cuda`가 나오도록 하는 것이 목표이다. (`print(torch.cuda.is_available())`가 `True`가 나오는 것과 같은 의미이다. )
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```


## 로컬 환경에서 GPU 사용하기
아래 과정을 따라오면 된다. 다만, 내 노트북에 GPU 칩이 탑재되어 있지 않다면 로컬 환경에서 GPU를 이용해 딥러닝 모델을 실행할 수 없다! (이런 경우에는 Colab과 같은 GPU 있는 서버에서 모델을 실행하자)
#### 1.GPU 확인하기
- 아래 방법을 통해 내 노트북에 NVIDIA GPU가 장착되어 있는지 확인한다.
- 확인 방법: **장치 관리자 열기 > 디스플레이 어댑터 클릭하기 > NVIDIA GPU가 목록에 있는지 확인하기**
- 디스플레이 어댑터를 클릭하면 시스템에 설치된 그래픽 카드 목록이 다음과 같이 표시된다. 해당 화면을 보면 "NVIDIA GeForce MX450"이 있기 때문에 노트북에 NVIDIA GPU가 장착되어 있다는 것을 알 수 있다. 정확히는  "NVIDIA GeForce MX450"이라는 모델이 설치되어 있다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/confirm_gpu.png)

#### 2. NVIDIA 드라이버 설치하기
- [NVIDIA 드라이버 download](https://www.nvidia.com/ko-kr/drivers/) 링크로 들어가서 앞서 확인한 내 GPU 정보와 운영체제를 입력하고 찾기 버튼을 누른 후, 설치를 진행하면 된다. (아래 첨부 그림은 `NVIDIA GeForce MX450` 기준으로 작성했다.)
- 내 windows 운영체제 버전 확인하는 방법: **설정 > 시스템 > 정보** 들어가서 장치 사양과 windows 사양을 확인하면 현재 사용하고 있는 윈도우 버전 및 비트 등을 알 수 있다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/driver.png){: .align-center}
- [GeForce Game Ready 드라이버 560.94 | Windows 11](https://www.nvidia.com/ko-kr/drivers/details/230886/) 드라이버를 설치했다. 이때 `NVIDIA 그래픽스 드라이버`로 선택하여 진행했다. (why? GPU를 사용하는 딥러닝 작업에 GeForce Experience는 불필요하기 때문에 시스템을 가볍게 하고자 기존 설정에서 변경을 했다.) 이외의 나머지는 기존에 선택된대로 두고 진행했다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/choose.png)

