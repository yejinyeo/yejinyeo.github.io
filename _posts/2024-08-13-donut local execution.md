---
title:  "1.Donut model 로컬에서 실행하기(local execution)"
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
#### 3. vscode에서 가상환경 적용하기
![]({{site.url}}/images/2024-08-14-Virtual Environments/vscode-myenv.png)


## Getting Started
#### 프로젝트 폴더 확인하기
- `misc`: README.md에 첨부된 이미지, 데모에 쓸 수 있는 샘플 이미지가 저장되어 있는 디렉토리
- `app.py`: Donut 모델을 간편하게 테스트할 수 있는 데모 파일이다. 모델이 어떻게 작동하는지 확인할 수 있다.

#### `app.py` 돌려보기
Donut 모델을 돌리기 위해 어떻게 해야하는지 너무 막막해서, 우선 `app.py`파일부터  코드를 분석하고 실행시켜보기로 했다.
##### 1) 주요 코드 분석
![]({{site.url}}/images/2024-08-13-donut local execution/app-주석.png)
##### 2) `app.py` 실행
vscode 상단 메뉴에서 `Terminal`-`New Terminal`를 누른 후, 아래에 띄어진 터미널 창에서 `Command Prompt`를 클릭하여 해당 프롬포트에서 실행하면 된다.
![]({{site.url}}/images/2024-08-13-donut local execution/command-prompt.png){: .img-width-half .align-center}
- Document Parsing task를 하기 위해 `CORD` dataset과 hugging face의 `donut-base-finetuned-cord-v2` trained model을 사용했다. 
```
python app.py --task cord --pretrained_path "naver-clova-ix/donut-base-finetuned-cord-v2"
```
- error 발생) `ModuleNotFoundError: No module named 'gradio'`가 발생하여, `gradio`를 설치하고 다시 실행시켰다.
```
pip install gradio
```
- error 발생) Donut model 로드 과정에서 모델이 사전 학습된 가중치를 로드할 때, 가중치 크기와 현재 모델의 구조 사이에 불일치로 인해 오류가 발생했다. `app.py` 파일의 코드를 다음과 같이 수정했다.
```
pretrained_model = DonutModel.from_pretrained(args.pretrained_path, ignore_mismatched_sizes=True)
```
- 다시 실행했을 때 local url이 뜨면 잘 작동한 것이다. local url을 복사하여 chrome에 접속하여 `misc`에 있는 샘플 영수증 이미지를 업로드했다. 그런데 결과가 error가 떴다. 이는 docvqa로 진행해도 동일한 error가 발생했다. `ignore_mismatched_sizes=True` 설정과 관련된 문제에서 비롯된 것 같다. 우선 해당 오류는 일단은 그냥 넘어가기로 했다.

![]({{site.url}}/images/2024-08-13-donut local execution/local-url.png){: .img-width-half .align-center}
![]({{site.url}}\images\2024-08-13-donut local execution\gradio-cord.png){: .img-width-half .align-center}
