---
title: "가상환경(Virtual Environments)"
categories: 
  - Python
tag:
  - 가상환경
  - anaconda
  - conda
---


## Introduction


## 가상환경(Virtual Environments)이란?


## conda를 이용한 가상환경
Anaconda Prompt를 열어서 진행하면 된다. 윈도우 검색창에 Anaconda Prompt를 검색하여 열면 된다.
#### 0. 프로젝트 디렉토리로 이동하기
- 작업할 프로젝트 디렉토리에 `requirements.txt` 파일(꼭 이 파일 이름이 아니더라도 패키지와 버전을 명시해둔 파일)이 있는 경우에 진행하는 단계이다. 해당 파일이 없는 경우에는 이 단계를 skip하고 다음 단계부터 진행해도 된다! ~~(근데 그냥 해놓는게 마음 편하다)~~ 
- `requirements.txt`란 python 프로젝트에서 필요한 패키지와 그 버전을 명시해둔 텍스트 파일이다. 이 파일은 프로젝트의 의존성을 관리하고, 다른 개발자들이 동일한 환경에서 작업할 수 있도록 돕기 위해 사용된다.
- 간혹 일부 프로젝트에서는 `requirements.txt` 대신 `setup.py`, `Pipfile`, 또는 `environment.yml` 파일을 사용하여 의존성을 관리할 수 있다. 그러므로 프로젝트에서 패키지와 버전 관리를 어느 파일에서 하는지 파악하는 게 선행되어야 한다.

##### 1) 파일탐색기에서 절대경로 복사하기
파일탐색기에서 작업할 프로젝트 디렉토리에 들어간 후, 주소창을 클릭한다. 그 다음 `ctr+c`를 통해 폴더의 절대경로를 복사하면 된다.

![]({{site.url}}/images/2024-08-14-Virtual Environments/copy-path.png)
##### 2) 복사한 경로로 이동하기
anaconda prompt에 다음과 같이 `cd` 명령어를 사용하여 내가 작업할 프로젝트 폴더로 이동한다.
```
cd [복사한 프로젝트 디렉토리 경로]
```
![]({{site.url}}/images/2024-08-14-Virtual Environments/cd.png)

`(base)`란 **conda 설치 시 제공되는 기본 환경**으로, conda prompt를 실행하면 기본적으로 `base` 환경이 활성화된다. (conda는 anaconda를 설치하면 자동으로 함께 설치된다.) 필요에 따라 새로운 가상환경을 생성해서 `base`환경과는 별도로 관리할 수 있다. 
{: .notice}


#### 1. 가상환경 생성하기
```
conda create -n [가상환경 이름] python=[python 버전] -y
```
- `-n`: name의 약자로, 가상환경의 이름을 설정하는 option이다. 
- `[가상환경 이름]`: 생성할 가상환경의 이름을 지정한다. 이름은 내가 원하는대로 기억하기 쉽게 지어주면 된다.
- `python=[python 버전]`: (생략 가능) 설치할 python 버전을 지정한다. 이 항목은 생략 가능하며, 생략 시 기본적으로 최신 버전의 python이 설치된다.
- `-y`: (생략 가능) 모든 확인 메시지를 자동으로 "Yes"로 처리하는 option이다. 보통 `conda create` 명령어를 실행하면 기본으로 설치될 패키지 목록이 나타나면서 "Proceed ([y]/n)?" 메시지가 나타난다. 이때 `-y` option을 추가하면 이 메시지를 생략하고 자동으로 진행되는 것이다.

예를 들어, ex1과 같이 입력하면 python 3.10 버전으로 myenv라는 이름의 가상환경을 생성한 것이다.
```python
# ex1) python 3.10 버전으로 myenv라는 이름의 가상환경 생성
conda create -n myenv python=3.10
# ex2) python 버전 지정하지 않고 가상환경 생성
conda create -n myenv
```

#### 2. 가상환경 활성화하기
파이썬 패키지 설치하기 전에 꼭 가상환경을 활성화를 해야한다. 만약 활성화를 하지 않은 채로 패키지를 설치한다면 내가 프로젝트를 위해 생성한 가상환경 공간이 아닌 `(base)`에 패키지가 설치된다. 그러니 이 단계를 까먹지 말자!
```
conda activate [가상환경 이름]
```
가상환경을 활성화하면 `(base)`가 `([가상환경 이름])`으로 바뀐 것을 확인할 수 있다.
![]({{site.url}}/images/2024-08-13-donut local execution/base to myenv.png)

#### 3. 패키지 설치하기
- 특정 패키지 설치
```
pip install [패키지 이름]
```
- 특정 버전의 패키지 설치
```
pip install [패키지 이름]==[버전]
```
- `requirements.txt` 파일을 통한 패키지 설치
```
pip install -r requirements.txt
```
- `setup.py`를 통한 패키지 설치
```
pip install .
```

#### 4. vscode에서 가상환경 적용하기
##### 1) `Python:Select interpreter` 실행하기
- vscode에서 프로젝트 폴더를 open한 다음 `ctrl+shift+p`를 누르고 `Python:Select interpreter`를 검색해 클릭한다.
- vscode창에서 아래 그림과 같이 오른쪽 아래를 누르는 방법도 있다!
![]({{site.url}}/images/2024-08-14-Virtual Environments/오른쪽 아래.png)

##### 2) 생성한 가상환경 선택하기
![]({{site.url}}/images/2024-08-14-Virtual Environments/vscode-myenv.png)