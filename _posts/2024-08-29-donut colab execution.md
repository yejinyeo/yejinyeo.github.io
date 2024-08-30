---
title:  "2.Donut model Colab에서 실행하기"
categories: 
  - 학부연구생
tag:
  - donut
  - colab
  - project
---

## Introduction



## 1. colab으로 donut model 가져오기
- [colab에서 github repository 다운로드 및 실행하기]()를 참고하여 donut model에 해당하는 github 폴더를 colab으로 clone한다.


## 2. 패키지 설치하기
- 아래 명령어를 실행시켜 `setup.py` 설정대로 패키지를 설치한다.
```python
!pip install .
```
- colab에서는 어차피 개별 노트북이기 때문에 가상환경을 굳이 만들 필요가 없다. (그러니 그냥 필요한 패키지 바로 설치하면 됨)


## 3. `train.py` 실행하기
아래 명령어를 실행시켰다.
```python
%run train.py --config ./config/train_cord.yaml --exp_version "test_experiment"
```
#### error1. 
- **오류 메시지**
    - 현재의 colab 환경(인터랙티브 환경)에서는 `Trainer(strategy='ddp')`가 호환되지 않는다는 것을 나타낸다.
    - colab과 같은 인터랙티브 환경에서는 `ddp`를 사용하려고 하면 아래와 같은 오류가 발생하기 때문에 colab에서 사용학 적합한 `ddp_notebook` 전략을 사용해야 한다.
    - **ddp(Distributed Data Parallel)**: 여러 GPU를 사용하는 분산 학습을 위한 전략 (스크립트로 실행될 때 더 잘 작동함)
```
---------------------------------------------------------------------------
MisconfigurationException                 Traceback (most recent call last)
/content/drive/MyDrive/github-project/donut/train.py in <module>
    174 
    175     save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
--> 176     train(config)

4 frames
/usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py in _lazy_init_strategy(self)
    578 
    579         if _IS_INTERACTIVE and self.strategy.launcher and not self.strategy.launcher.is_interactive_compatible:
--> 580             raise MisconfigurationException(
    581                 f"`Trainer(strategy={self._strategy_flag!r})` is not compatible with an interactive"
    582                 " environment. Run your code as a script, or choose a notebook-compatible strategy:"

MisconfigurationException: `Trainer(strategy='ddp')` is not compatible with an interactive environment. Run your code as a script, or choose a notebook-compatible strategy: `Trainer(strategy='ddp_notebook')`. In case you are spawning processes yourself, make sure to include the Trainer creation inside the worker function.
```
- **solution: `train.py` 파일 수정하기**
    - `train.py`에서 `Trainer`를 생성하는 부분의 `strategy="ddp"`를 `strategy="ddp_notebook"`로 변경한다.