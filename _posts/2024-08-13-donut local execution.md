---
title:  "1.Donut model 로컬 실행(local execution)"
categories: 
  - 학부연구생
tag:
  - donut
  - 가상환경
---

## Introduction
"**OCR-free Document Understanding Transformer (ECCV 2022)**"를 연구의 참고 논문으로 선정했다. 본격적인 연구에 앞서 [donut github](https://github.com/clovaai/donut?tab=readme-ov-file)의 `README.md`를 기반으로 논문에서 제안한 Donut model을 local에서 돌려보고자 한다.


## Setup
#### 1. gitbub repository 복사하기
##### 1.1. `clovaai/donut` repository를 fork하기
[[fork: 다른 사람의 repository를 내 repository로 복사하기]](https://yejinyeo.github.io/git&github/fork/)를 참고해서 fork를 한다.

![yeinyeo-donut]({{site.url}}/images/2024-08-13-donut local execution/yejinyeo-donut.png)

##### 1.2. 복사한 repository를 local에 clone하기
[[clone: repository를 내 컴퓨터(local)로 복제하기]](https://yejinyeo.github.io/git&github/clone/)를 참고해서 내 계정에 있는 repository를 local에 clone 하고, vscode에 open한다.

#### 2. conda를 이용해 가상환경 설정하기
[[가상환경(Virtual Environments)]](https://yejinyeo.github.io/python/Virtual-Environments/)의 "conda를 이용한 가상환경" 부분을 참고하며, anaconda prompt에서 진행하면 된다.
##### 2.1. 프로젝트 디렉토리로 이동하기
```
cd [프로젝트 디렉토리 경로]
```
![]({{site.url}}/images/2024-08-14-Virtual Environments/cd.png)
##### 2.2. 가상환경 생성하기
```
conda create -n donut_official python=3.7
```
##### 2.3. 가상환경 활성화하기
```
conda activate donut_official
```
![]({{site.url}}/images/2024-08-13-donut local execution/base to myenv.png)
##### 2.4. 패키지 설치하기
`pip install .`는 `setup.py` 파일을 기반으로 현재 디렉토리의 패키지를 설치하기 위한 명령어이다. `setup.py`에서 정의된 설정에 따라 패키지를 설치하기 위해 다음과 같은 명령어를 입력하면 된다.
```
pip install .
```
##### 2.5. vscode에서 가상환경 적용하기
![]({{site.url}}/images/2024-08-14-Virtual Environments/vscode-myenv.png)


