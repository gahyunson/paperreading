# VGGNet paper Review

Click HERE to Move paper : [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf) 

<aside>
💡 This paper evaluates networks with increasing depth using an architecture that employs small 3x3 convolution filters, resulting in a depth of 16-19 weight layers. The use of small filters allows for stable increase in network depth by adding more convolutional layers.

</aside>

## CONVNET CONFIGURATIONS

### 1. **Architecture**

| input convNets | image data - 224 x 224 RGB 고정 |
| --- | --- |
| input image preprocess | (training image data - RGB 평균값)  |
| input filters | 1 x 1 (input image channels를 선형 변환함) |
| receptive field | 3 x 3 convolution layers  |
| convolution | stride : 1px 고정 |
| hidden layers의 activation | non-linearity(ReLU) |
| padding | 1px to 3 x 3 convolutional layers |
| pool layers | max pooling layers 5개 |
|  | 2 x 2 filter , stride : 2 |
| convolutional layers | Fully-Connected layers 3개 |
|  | 4096 channels + 4096 channels + 1000 channels |
| final layer | the soft-max layer |

receptive field와 input filter의 차이는 무엇일까?

- receptive field : input data를 보는 크기
- input filter : input data를 전처리하는 필터

### 2. Configurations

논문의 Table 1에 A to E 열을 이용하여 nets에 대한 정보를 제공한다. 

범위는 A 네트워크에서는 11개 레이어, E 네트워크에서는 19개 레이어이다. 뉴런의 범위는 64개 ~ 512개이다. Table2 에서는 파라미터 값을볼 수 있다. 해당 테이블만 봐두 구현이 가능할 듯 하다.

### 3. Discussion

- 3x3 filter 3개 = 7x7 filter 1개
- 파라미터 수가 더 적다 = 연산량이 더 적다 = 빠르게 결과 도출 가능

예시) C channels를 가진다고 가정할 때 3x3 conv stack 3개의 파라미터 값은 27$C^2$이고 7x7 1개의 파라미터 값은 49$C^2$이다. 

- 1x1 conv.layers : 비선형성 증가 (receptive field의 영향없이)
- 7x7 보다 3x3 필터가 더 좋은 이유는? 더 작은 필터를 사용하면 더 많은 비선형성 결과를 가질 수 있다. 이미지의 더 많은 특징을 추출할 수 있게 된다.

---

## Classification Framework

### 1. Training

|  | mini-batch gradient descent with momentum (multinomial logistic regression) |
| --- | --- |
| batch size | 256 |
| momentum | 0.9 |
| regularisation | L2 penalty |
| dropout | 0.5 (1st, 2nd fc layers) |
| learning rate | 초기값 : $10^{-2}$
validation accuracy 증가가 멈출 때마다 0.01(10배씩) decrease
결과 → 3 times 감소하게 됨, 370K(74 epochs) 반복해서 멈춤 |
| input image | 224x224 고정
+ 랜덤으로 이미지를 rescale하여 224 크기로 자름,
+ 자른 이미지를 수평으로 뒤집거나 랜덤하게 RGB 값을 변경 |
|  | 384 크기의 이미지로 사전훈련된 a single-scale model 의 fine-tuning → configuration으로 multi-scale models를 훈련 |

> 논문의 저자들은 많은 파라미터와 깊은 깊이를 가진 nets가 `implicit regularisation`와 특정층의 레이어 초기값으로 인해서 적은 epochs로 수렴된다 짐작했다. The initialisation of the network weights이 중요하다 판단함.
> 

→ 문제점 : 깊은 nets에서 나쁜 초기값은 학습을 멈추게 하거나 gradient를 불안정하게 할 수 있다.

→ 해결 방법 : Table 1의 상대적으로 적은 레이어인 A configuration 으로 랜덤 초기값으로 훈련 반복(learning rate 감소안함) → training architectures가 더 깊어졌을 때 해당 값으로 처음 4개 conv layers와 마지막 fc layers 3개 초기값 설정(나머지는 랜덤 , 범위는 0 ~ $10^{-2}$), biases : 0로 초기화. 

→ 결론 : 타 논문을 통해 pre-training 없이도 초기값 설정이 가능함을 알아냄.

### 2. Testing

| test input scale | train input scale이 같지 않아도 된다 |
| --- | --- |
|  | horizontal flipping 이미지 사용 |
| the fully-convolutional network | 앞서 3개의 fc layer를 test 할 때 아래와 같이 바꿈
1st fc layer → 7x7 conv. layer
2nd, 3rd fc layer → 1x1 conv. layer |
|  | 자르지 않은 전체 이미지가 적용됨(conv.로만 이루어졌기 때문, 따로 이미지를 잘라서 input값으로 넣는 작업을 안해도 된다) |
| final layer 결과 (softmax 직전) | a class score map size = 이미지 크기가 같으면 늘 같은 값이 나옴 |
| final score | = spatially averaged = original images와 flipped images 의 평균값 |
| multi-crop evaluation  | 서로 다른 conv. 경계 조건을 가져서 dense evaluation과 상호보완적인 것 |
|  | ConvNet 을 a crop에 적용 → paddig : 0 |
| dense evaluation | 0 padding이 아닌 crop 한 주변의 부분들을 padding 값으로 사용 |
|  | Receptive field 크기 증가 |

### 3. Implementation details

- the publicly available C++ Caffe toolbox 을 기반으로 구현됨
- 하나의 시스템에 설치된 여러 개의 GPU에서 train, evaluate + 다양한 crop하지 않은 이미지 학습 및 평가 수행
- GPU가 병렬 처리하여도 GPU 배치 기울기(전체 배치 평균 기울기 적용)가 동일하게 적용되므로 단일 GPU와 정확히 동일한 결과 추출 가능
- a single GPU 속도 < an off-the-shelf 4-GPU (4개의 GPU 장착 시스템)  [3.75배 더 빠름]
- NVIDIA Titan Black GPUs로 2~3주 학습 시간 소요

---

## Classification experiments

| Class | 1000개 |
| --- | --- |
| train image |  1.3M |
| validation image | 50K |
| test image | 100K |

(validation set을 test set으로 사용함)

분류 성능은 두 가지 기준(top-1, top-5 에러율)으로 측정됐다. 

- top-1 : multi-class classification error 잘못 분류된 이미지 비율
- top-5 : 상위 5개 예측 카테고리 밖의 이미지 비율 (main evaluation 주요 평가 기준 criterion used in ILSVRC)

### 1. Single scale evaluation

scale 값을 두 가지(256, 384) 고정하고 훈련 

256x256 : 사전 훈련용, 이후 384 훈련에 사전 훈련 모델로 사용됨, lr: $10^{-3}$

384x384 : 256 사전 훈련을 사용하기 때문에 속도가 더 빠름, 256 훈련시 나온 가중치 사용, 더 작은 learning rate 값 사용

### 2. Multi-scale evaluation

S 값을 multi-scale 을 통해서 설정한다

256 ~ 512 사이의 S로 랜덤 샘플링하여 훈련 (=scale jittering)

속도의 이유로 앞서 384로 훈련한 동일 configuration의 single scale model의 모든 레이어를  fine-tuning하여 multi-scale models를 훈련

### 3. Multi-crop evaluation

**dense Conv evaluation** vs **mult-crop evaluation**

상단 두 가지 evaluation의 평균으로 softmax output을 추출

### 4. ConvNet fusion

몇 개의 모델들의 output evaluation을 평균내는 것으로 성능을 높임.

single-scale networks , a multi-scale model (fully-connected 부분만 fine-tuning 한 모델) 으로만 훈련.

→ test error : 7.3% (7개 networks 앙상블 모델)

→ 가장 성능이 좋은 multi-sclae models 2개로 앙상블 시도

7.3% 에러로 2등을 했다. 앙상블 2개 모델을 이용한 결과로는 6.8% 에러율이 나왔다. (D/[256;512]/256,384,512), (E/[256;512]/256,384,512) 

→ test error : 7.0% (with dense)

→ test error : 6.8% (with dense & multi-crop)

→ test error : 7.1% (with single model)

(dense & multi-crop의 조합이 가장 좋은 것을 볼 수 있다)

## Conclusion

이미지 분류를 위한 최대 19개 층의 convolutional networks를 만들었다. 큰 데이터셋임에도 convolutional networks로 깊은 레이어 설계가 가능한 것을 보여줬고, 깊이가 중요성을 알려준다.