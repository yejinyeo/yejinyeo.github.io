---
layout: single
title:  "Perceptron"
categories: Deep-Learning
---

# **1. Perceptron**
*여기서 기술하는 퍼셉트론은 단순 퍼셉트론임*

#### **퍼셉트론**
신경망(딥러닝)의 기원이 되는 알고리즘  
**다수의 신호($x_{1}$, $x_{2}$)를 입력으로 받아 하나의 신호($y$)를 출력**  

![퍼셉트론](https://velog.velcdn.com/images%2Fjakeseo_me%2Fpost%2Fc120731c-21dd-4eec-8ee3-58b6e6f52d05%2Fperceptron.jpg)

[그림1-1] 입력이 2개인 퍼셉트론

- 입력 신호: $x_{1}$, $x_{2}$
- 출력 신호: $y$  
- 가중치: $w_{1}$, $w_{2}$  
- (그림의) 원: 뉴런 or 노드

- 퍼셉트론 신호 값:  
        1 (신호가 흐른다)  
        0 (신호가 안 흐른다)
- 각각의 입력 신호에 고유한 **가중치**가 할당되어 곱해짐($w_{1}x_{1}$, $w_{2}x_{2}$)
- **가중치**: 각 신호가 결과에 주는 영향력 조절하는 요소, 중요도(가중치가 클수록 해당 신호가 그만큼 더 중요함)
- 뉴런에서 보내온 신호의 총합($w_{1}x_{1} + w_{2}x_{2}$)이 정해진 한계(**임계값**, $\theta$)을 넘어설 때만 1을 출력  
  -> '뉴런이 활성화한다'
#### **퍼셉트론 동작 원리의 수학적 표현**
$y =
\begin{cases}
1,\;if\;w_{1}x_{1} + w_{2}x_{2} > \theta \\
0,\;otherwise\;(w_{1}x_{1} + w_{2}x_{2} \leq \theta)
\end{cases}$

# **2. 단순 논리 회로**
### **2.1 퍼셉트론으로 AND, NAND, OR 논리 회로를 표현해보자.**
##### **how?** 진리표대로 작동하도록 하는 $w_{1},\;w_{2},\;\theta$ 값 설정하기
- 진리표: 입력 신호와 출력 신호의 대응 표
1. AND 게이트
2. NAND 게이트
3. OR 게이트
- 입력: 2개  
- 출력: 1개  

#### **AND 게이트**
**두 입력이 모두 1일 때만 1을 출력, 그 외에는 0을 출력** 

($w_{1}$, $w_{2}$, $\theta$) = (0.5, 0.5, 0.7) or (0.5, 0.5, 0.8) or (0.5, 0.5, 1.0)  
-> [그림2-1] 진리표를 만족하는 매개변수 조합은 무수히 많음.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbi03NF%2FbtsDofZBfAI%2FjkPPmc0s7K52Eb9X9Mx5aK%2Fimg.png" width="300" height="300"/>

[그림2-1] AND 게이트의 진리표  

#### **NAND 게이트**
NAND(Not AND): **AND 게이트의 출력을 뒤집은 것**  
**두 입력이 모두 1일 때만 0을 출력, 그 외에는 1을 출력** 

($w_{1}$, $w_{2}$, $\theta$) = (-0.5, -0.5, -0.7) or (-0.5, -0.5, -0.8) or (-0.5, -0.5, -1.0)  
-> AND 게이트를 구현하는 매개변수의 부호를 모두 반전하면 됨.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqFYJ9%2FbtsDpykcB1B%2FWALeGzKJuUBkiEvZ1kK3l0%2Fimg.png" width="300" height="300"/>

[그림2-2] NAND 게이트의 진리표  

#### **OR 게이트**
**입력 신호 중 하나 이상이 1일 때 1을 출력, 그 외(입력 신호가 모두 0)에는 0을 출력** 

($w_{1}$, $w_{2}$, $\theta$) = (0.5, 0.5, 0.4)  

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdm0sc1%2FbtsDp3K6flk%2FzQIoLim9iy4HooM8iv94P1%2Fimg.png" width="300" height="300"/>

[그림2-3] OR 게이트의 진리표  

### **2.2 학습이란?**
**최적의 매개변수 값을 설정하는 작업**

#### **in 퍼셉트론으로 논리회로 표현**
퍼셉트론의 구조는 모든 게이트에서 동일함, 매개변수(가중치와 임계값)의 값만 다름.  
우리(사람)가 직접 학습 데이터를 보며 매개변수 값을 설정함.
- 학습 데이터: 진리표
- 퍼셉트론의 매개변수 값을 정하는 주체: 사람 (not 컴퓨터)

#### **in 기계학습**
컴퓨터가 자동으로 매개변수 값을 설정함.  
사람: 퍼셉트론의 구조(모델) 결정, 학습 데이터 컴퓨터에게 제공
##### **Q. 기계학습이 딥러닝과 무슨 관계인가? [(딥러닝과 머신러닝의 차이점)](https://mvje.tistory.com/138)**
A. 딥러닝은 기계학습의 한 종류이다!  
딥러닝: 인공신경망(Artificial Neural Network, ANN)을 사용하여 대규모의 데이터를 학습하는 기계학습(Machine Learning)의 한 분야

# **3. 퍼셉트론 구현하기**

### **3.1 논리회로 구현**


```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7    # 매개변수 초기화
    tmp = x1*w1 + x2*w2
    if tmp > theta:
        return 1
    else:
        return 0
```


```python
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
```

    0
    0
    0
    1
    

### **3.2 가중치와 편향 도입**
#### **$\theta$를 $-b$로 치환하자.**
$y =
\begin{cases}
1,\;if\;w_{1}x_{1} + w_{2}x_{2} + b > 0 \\
0,\;otherwise\;(w_{1}x_{1} + w_{2}x_{2} + b \leq 0)
\end{cases}$  
- $b$ : **편향(bias)**
- $w_{1}, w_{2}$ : 가중치
> **퍼셉트론**: 입력 신호에 가중치를 곱한 값과 편향을 합하여, 그 값이 0을 넘으면 1, 그렇지 않으면 0을 출력함.


```python
# 'numpy를 이용한' 퍼셉트론 식 구현
import numpy as np
x = np.array([0, 1])    # 입력
w = np.array([0.5, 0.5])    # 가중치
b = -0.7    # 편향
```

**element-wise product(원소별 곱셈)**: index가 같은 원소끼리 곱함.


```python
w*x
```




    array([0. , 0.5])



**np.sum()**: 입력한 배열에 담긴 모든 원소의 총합 계산


```python
np.sum(w*x)
```




    0.5




```python
np.sum(w*x) + b
```




    -0.19999999999999996



$w*x + b = -0.19999999999999996 \leq 0$  
이므로 $y = 0$임.

### **3.3 가중치와 편향 구현**
- $-\theta$가 편향 $b$로 치환
- **가중치**: 각 입력 신호가 결과에 주는 영향력(중요도) 조절하는 매개변수
- **편향**: 뉴런이 얼마나 쉽게 활성화되는지(결과로 1을 출력)를 조정하는 매개변수  
        ex1) $b = -0.1$ -> 각 입력 신호에 가중치를 곱한 값들의 합이 0.1를 초과할 때만 뉴런 활성화됨.  
        ex2) $b = -20.0$ -> 각 입력 신호에 가중치를 곱한 값들의 합이 20.0을 초과할 때만 뉴런 활성화됨.

#### **'가중치와 편향 도입한' AND 게이트**
($w_{1}$, $w_{2}$, $\theta$) = (0.5, 0.5, 0.7) 사용  
주의) $b = -\theta$


```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0
```

**NAND, OR 게이트 구현**: w, b(매개변수 값)만 바꿔주면 됨. 

**AND, NAND, OR 게이트의 공통점 및 차이점**  
공통점: 퍼셉트론의 구조  
차이점: w, b(매개변수 값)   

ex)  
NAND 게이트: ($w_{1}$, $w_{2}$, $\theta$) = (-0.5, -0.5, -0.7)  
OR 게이트: ($w_{1}$, $w_{2}$, $\theta$) = (0.5, 0.5, 0.4)


```python
# NAND 게이트 구현
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0
```


```python
# OR 게이트 구현
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(w*x) + b
    if tmp > 0:
        return 1
    else:
        return 0
```

# **4. 퍼셉트론의 한계**
- **단층 퍼셉트론(single-layer perceptron)**
으로는 XOR 게이트를 표현할 수 없다.
- **단층 퍼셉트론은 비선형 영역을 분리할 수 없다.**

### **4.1 XOR 게이트**
XOR 게이트(배타적 논리합): **$x_1, x_2$ 중 한쪽이 1일 때만 1을 출력**  
퍼셉트론의 시각화에서 출력값이 0일 때와 1일 때를 **직선**으로 나누는 것 불가능함.

### **4.2 선형과 비선형**
- **비선형 영역**: 곡선의 영역  
- **선형 영역**: 직선의 영역  
if 직선 제약을 없애면 -> 곡선으로 나눌 수 있음.  
#### **but 퍼셉트론의 한계** 
**직선 하나로 나눈 영역만 표현 가능함.** -> 곡선 표현 불가능

# **5. 다층 퍼셉트론(multi-layer perceptron)**
다층 퍼셉트론을 사용하여 XOR 게이트 문제를 해결해보자.  
#### **다층 퍼셉트론**
**층이 여러 개인 퍼셉트론**

### **5.1 기존 게이트 조합하기**
입력 신호 $x_1, x_2$와 출력 신호 $y$가 있을 때,  
**$x_1, x_2$는 NAND와 OR 게이트의 입력**이 되고, **NAND와 OR 게이트의 출력이 AND 게이트의 입력**으로 이어지도록 **AND, NAND, OR 게이트를 조합**하기

### **5.2 XOR 게이트 구현하기**


```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```


```python
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
```

    0
    1
    1
    0
    

#### **XOR의 다층 퍼셉트론**
- AND, OR: 단층 퍼셉트론
- XOR: 다층 퍼셉트론(2층 퍼셉트론)  
  *구성 층의 수를 기준으로 '3층 퍼셉트론'이라 하는 경우도 있음*

![XOR의 퍼셉트론](data:image/png;base64,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)

[그림 5-1] XOR의 퍼셉트론  
1. 0층의 두 뉴런이 입력 신호를 받아 1층의 뉴런으로 신호 보냄.
2. 1층의 두 뉴런이 신호를 받아 2층의 뉴런으로 신호 보냄.
3. 2층의 뉴런이 받은 신호를 바탕으로 y를 출력함.

> 단층 퍼셉트론으로는 표현하지 못한 것을 층을 쌓아(층을 늘려, 깊게 하여->**다층 퍼셉트론**) 더 다양한 것을 표현할 수 있다.

# **6. 퍼셉트론으로 컴퓨터 표현**

#### **다층 퍼셉트론(이론상 2층 퍼셉트론)으로 컴퓨터를 만들 수 있다!**
- NAND 게이트의 조합만으로 컴퓨터를 만들 수 있다.
- NAND 게이트는 퍼셉트론으로 만들 수 있다.

> **퍼셉트론**  
> 층을 거듭 쌓으면 비선형적인 표현도 가능하고, 이론상 컴퓨터가 수행하는 처리도 모두 표현할 수 있다.

# **7. 요약**
- 퍼셉트론은 입출력을 갖춘 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다.
- 퍼셉트론에서는 **가중치**와 **편향**을 매개변수로 설정한다.  
  *내가(사람) 데이터(진리표)를 보고 직접 설정했음*
- 퍼셉트론으로 AND, NAND, OR 게이트 등의 논리 회로를 표현할 수 있다.
- XOR 게이트는 단층 퍼셉트론으로는 표현 할 수 없다.
- 2층 퍼셉트론(다층 퍼셉트론)을 이용하면 XOR 게이트를 표현할 수 있다.
- **단층 퍼셉트론으로는 직선형 영역만 표현 가능, 다층 퍼셉트론은 비선형 영역도 표현 가능**
- 다층 퍼셉트론은 (이론상) 컴퓨터를 표현할 수 있다.
