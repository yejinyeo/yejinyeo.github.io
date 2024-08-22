---
title: "로컬에서 딥러닝을 위한 GPU 개발환경 구축하기"
categories: 
  - Deep-Learning
tag:
  - GPU
  - CUDA
---


## Introduction
로컬 환경에서 vscode를 사용해 딥러닝 모델을 실행하려고 할 때, GPU 설정이 필요하다. Colab과 같이 GPU가 있는 서버 환경에서는 GPU 설정이 자동으로 이루어져 복잡한 과정 없이 바로 사용할 수 있지만, 로컬에서는 직접 설정을 해줘야 한다. 과정이 꽤 복잡하기 때문에 로컬에서 딥러닝을 위한 GPU 개발환경을 구축 및 설정하는 방법을 정리했다.


## Goal
vscode의 terminal(cmd)에서 위 코드를 실행했을 때, 결과가 `cuda`가 나오도록 하는 것이 목표이다. (`print(torch.cuda.is_available())`가 `True`가 나오는 것과 같은 의미이다. )
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```


## 로컬에서 GPU 개발환경 구축하기
아래 과정을 따라오면 된다. 다만, 내 노트북에 GPU 칩이 탑재되어 있지 않다면 로컬 환경에서 GPU를 이용해 딥러닝 모델을 실행할 수 없다! (이런 경우에는 Colab과 같은 GPU 있는 서버에서 모델을 실행하자)
#### 1.GPU 확인하기
- 아래 방법을 통해 내 노트북에 NVIDIA GPU가 장착되어 있는지 확인한다.
- 확인 방법: **장치 관리자 열기 > 디스플레이 어댑터 클릭하기 > NVIDIA GPU가 목록에 있는지 확인하기**
- 디스플레이 어댑터를 클릭하면 시스템에 설치된 그래픽 카드 목록이 다음과 같이 표시된다. 해당 화면을 보면 "NVIDIA GeForce MX450"이 있기 때문에 노트북에 NVIDIA GPU가 장착되어 있다는 것을 알 수 있다. 정확히는  "NVIDIA GeForce MX450"이라는 모델이 설치되어 있다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/confirm_gpu.png)

#### 2. NVIDIA 드라이버 설치하기
- [NVIDIA 드라이버 설치하러가기](https://www.nvidia.com/ko-kr/drivers/)
- 위 링크로 들어가서 앞서 확인한 내 GPU 정보와 운영체제를 입력하고 찾기 버튼을 누른 후, 설치를 진행하면 된다. (아래 첨부 그림은 `NVIDIA GeForce MX450` 기준으로 작성했다.)
- 내 windows 운영체제 버전 확인하는 방법: **설정 > 시스템 > 정보** 들어가서 장치 사양과 windows 사양을 확인하면 현재 사용하고 있는 윈도우 버전 및 비트 등을 알 수 있다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/driver.png){: .align-center}
- [GeForce Game Ready 드라이버 560.94 | Windows 11](https://www.nvidia.com/ko-kr/drivers/details/230886/) 드라이버를 설치했다. (탑재된 GPU마다 설치해야하는 드라이버가 다르니 무작정 이 링크 들어가서 설치하면 안된다!) 설치 후 다운로드된 프로그램 파일을 클릭하여 실행하면 완료가 된다. 이때 `NVIDIA 그래픽스 드라이버`로 선택하여 진행했다. (why? GPU를 사용하는 딥러닝 작업에 GeForce Experience는 불필요하기 때문에 시스템을 가볍게 하고자 기존 설정에서 변경을 했다.) 이외의 나머지는 기존에 선택된대로 두고 진행했다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/choose.png)
- 정상 설치 여부 확인 방법: cmd 창에서 아래 명령어 입력한 후, 출력 결과가 아래 그림과 같이 나오면 정상적으로 설치된 것이다. 출력 결과에서 Driver Version과 CUDA Version는 확인하고 기억해두자!
```
nvidia-smi
```
![]({{site.url}}/images/2024-08-21-GPU setting in local/nvidia.png)

#### 3. Cuda Toolkit 설치하기
여기서부터 설치가 까다로워진다. 버전을 제대로 확인하고 설치해야하기 때문이다. 
##### 3.1 내 GPU와 맞는 CUDA version 찾기
**1) 내 GPU의 compute capability 찾기**
- 내 GPU(`NVIDIA GeForce MX450`)와 호환되는 Cuda Toolkit version이 무엇인지 확인하기 위해서는 먼저 내 GPU의 **compute capability**를 찾아야 한다. [[Your GPU Compute Capability]](https://developer.nvidia.com/cuda-gpus)에서 `CUDA-Enabled GeForce and TITAN Products`를 클릭해서 내 GPU 이름에 해당하는 Compute Capability를 찾으면 된다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/nvidia compute.png){: .align_center}
- 만약 위 링크에서 내 GPU 이름이 없다면 [[wikipedia CUDA]](https://en.wikipedia.org/wiki/CUDA)의 `GPUs supported` 파트에 있는 `Compute Capability, GPU semiconductors and Nvidia GPU board products` 표에서 내 GPU 이름을 찾는다. (나는 `GeForce` 열에서 찾았음) 그다음 찾은 행에서 `Compute capability(version)` 열에 해당하는 값을 확인한다.(내 GPU는 7.5d였음)

![]({{site.url}}/images/2024-08-21-GPU setting in local/index.png)
![]({{site.url}}/images/2024-08-21-GPU setting in local/compute.png)

**2) compute capability에 호환되는 CUDA version 찾기**
- [[wikipedia CUDA]](https://en.wikipedia.org/wiki/CUDA)의 `GPUs supported` 파트에서 `Compute Capability (CUDA SDK support vs. Microarchitecture)` 표를 참고하여 앞서 찾은 compute capability를 포함하는 `CUDA SDK Version(s)` 열에 해당하는 값이 호환되는 CUDA 버전이다. (아래 첨부한 표를 참고해서 찾으면 됨!)
- 표 보는 방법: 연두색 칸에서 나의 compute capability가 해당되는 행의 `CUDA SDK Version(s)` 열 값이 모두 호환되는 CUDA version이다. 예를 들어 compute capability가 7.5이면 10.0~12.5 까지의 CUDA SDK version을 사용할 수 있는 것이다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/cuda version.png)

##### 3.2 Cuda Toolkit 설치하기
- [Cuda Toolkit 설치하러가기](https://developer.nvidia.com/cuda-toolkit-archive)
- 위 링크로 들어가서 앞서 찾은 version에 해당하는 Cuda Toolkit을 설치하면 된다. (https://developer.nvidia.com/cuda-downloads 이 링크로 접속 시, 현 상황에 맞는 버전으로 자동으로 연결된다고 한다! ~~확실치는 않지만 나 같은 경우는 최신 버전인 12.6의 compute capability를 몰라서 12.5와 12.6을 고민하다가 이 링크로 들어가 12.6으로 설치를 진행했다.~~)
- 설치 전 setting: 본인의 os에 맞게 선택하면 된다. 다만, `Installer Type`는 꼭 `exe(local)`로 하는 것이 좋다. (why? local 타입은 설치에 필요한 모든 파일을 exe 형태로 제공함. network 타입은 최소한의 파일만 제공하고 네트워크를 통해 설치하는 것이기 때문에 local 타입보다 오류 발생 가능성이 높음.)
![]({{site.url}}/images/2024-08-21-GPU setting in local/cuda setting.png)
- CUDA Toolkit 설치시 기본적으로 설정된 것 그대로 안건드리고 진행하면 된다. (설치 시 처음으로 나오는 경로 설정창에서도 경로 그대로 두면 됨. 내 경우에는 Extraction path: `C:\Users\yedin\AppData\Local\Temp\cuda`였음.) 마지막 설정창에서 체크 표시 중 아래 체크 표시는 없애고 설치를 마쳤다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/final check.png)
- 위의 과정대로 설치를 진행했다면(Cuda Toolkit이 설치된 경로에 다른 설정을 하지 않은 한) Cuda Toolkit 경로는 다음과 같은 형식일 것이다. 해당 경로는 이후 단계에서 필요하니 기억해두자!

**Cuda Toolkit 경로**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v버전`   
ex) 내 Cuda Toolkit 경로: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
{: .notice--success}
- 정상 설치 여부 확인 방법: cmd창에서 다음 명령어를 입력했을 때 아래처럼 출력이 나오면 정상적으로 설치된 것이다.
```
nvcc -V
```
![]({{site.url}}/images/2024-08-21-GPU setting in local/nvcc.png)


#### 4. cuDNN 설치하기
##### 4.1 NVIDIA Developer 로그인하기
- [NVIDIA Developer 로그인하러 가기](https://developer.nvidia.com/)
- cuDNN을 설치하기 위해서는 위 링크에 접속해서 NVIDIA Developer에 회원가입하고 로그인해야한다.

##### 4.2 cuDNN 설치하기
- [cuDNN 설치하러 가기(version 선택 가능)](https://developer.nvidia.com/rdp/cudnn-archive)
- [cuDNN 설치하러 가기(version 선택 없이 설치)](https://developer.nvidia.com/cudnn-downloads)
- 첫번째 링크로 들어가는 경우: 앞서 설치한 CUDA 버전에 해당하는 cuDNN을 설치하면 된다. (나는 이 방법으로 설치를 진행함)
- 두번째 링크로 들어가는 경우: 링크로 들어가서 설치하면 된다. (처음에 이 링크로 시도했지만 windows 11 선택창이 없어서 첫번째 링크로 들어감)
- 앞서 설치한 내 CUDA 버전(12.6)에 해당하는 cuDNN을 선택하여 설치하면 된다.  
나는 CUDA 12.6이기 때문에 가장 첫번째 cuDNN을 클릭했다. 
![]({{site.url}}/images/2024-08-21-GPU setting in local/cu start.png)  
클릭해서 나오는 목록에서 첫번째 Windows용 Zip 파일을 클릭하여 다운받으면 된다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/cu choose.png)


#### 5. cuDNN 압축 파일 해제 및 Cuda Toolkit 폴더에 파일 복사하기
##### 5.1 cuDNN 압축 파일 해제하기
- 설치한 cuDNN 파일을 압축 해제하면 다음과 같은 파일들로 구성되어 있음을 확인할 수 있다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/file config.png)

##### 5.2 압축 해제한 파일 복붙하기
- 압축 해제한 cuDNN 폴더 내 `bin`, `include`, `lib` 디렉토리들을 복사해서 Cuda Toolkit 폴더(앞서 언급했던 Cuda Toolkit 경로: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v버전`) 안에 붙여넣기한다. 동일한 폴더 이름이 있더라도 폴더 내 파일이 다르기 때문에 같은 이름의 폴더 안으로 파일들이 들어가니 신경 안써도 된다. (그냥 복붙하면 됨!)
![]({{site.url}}/images/2024-08-21-GPU setting in local/cuda toolkit.png)
- 복붙하는 이유: cuDNN의 파일들을 CUDA 파일들과 함께 보관함으로써, 환경변수를 통해 CUDA를 불러올 때 cuDNN의 파일을 같이 불러올 수 있게 하기 위함이다.


#### 6. 환경변수 확인하기
- 마지막으로 CUDA를 활용하기 위해 PATH를 확인하면 된다. CUDA 설치 시 기본적으로 windows의 환경 변수(PATH)에 등록이 되지만, 확인을 위해 아래 과정을 진행한다.
- 환경변수 확인하는 방법: **`고급 시스템 설정 보기` 시작창에 입력 후 열기 > 하단의 `환경 변수` 버튼 클릭하기
- 아래와 같이 `CUDA_PATH`와 `CUDA_PATH_V버전` 이렇게 두 개가 뜨면 정상이다.
![]({{site.url}}/images/2024-08-21-GPU setting in local/env var.png)