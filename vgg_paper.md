# VGG

# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

### ABSTRACT

a thorough evaluation of networks of increasing depth using an architecture with very small 3x3 conv filters and pushing the depth to 16-19 weight layers. 

### INTRODUCTION

Steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small (
3×3) convolution filters in all layers.

### CONVNET CONFIGURATIONS

1. Architecture

input convNets : 224 x 224 RGB 이미지로 고정되어 있다. 

receptive field : 3 x 3 의 convolution layers 

input image preprocess : training image data 에서 RGB 평균값을 뺀다.

input filters : 1 x 1 (input image channels를 선형 변환함)

convolution - stride : 1 px로 고정

padding  : 3 x 3 convolutional layers를 사용하였다

pool layers : 5개의 maxpooling layers

window는 2 x 2 pixel , stride는 2

hidden layers의 activation : non-linearity(ReLU) 사용

convolutional layers : 3개의 Fully-Connected layers.

4096 channels + 4096 channels + 1000 channels

final layer : the soft-max layer.

1. Configurations

Table 1에 A to E 열을 이용하여 nets에 대한 정보를 제공한다. 범위는 A 네트워크에서는 11개 레이어, E 네트워크에서는 19개 레이어가 존재한다. 뉴런의 수도 64개에서 512개를 사용하기도 했다. Table2 에서는 파라미터 값을볼 수 있다.

1. Discussion

예를 들어 C channels를 가진다고 가정할 때 3x3 conv stack 3개의 파라미터 값은 27$C^2$이고 7x7 1개의 파라미터 값은 49$C^2$이다. 

1x1 conv.layers : 비선형성 증가 (receptive field의 영향없이)

### Classification Framework

1. Training

mini-batch gradient descent - momentum을 이용한 multinomial logistic regression으로 훈련했다. 

batch size : 256

momentum : 0.9

regularisation : L2 penalty , dropout(0.5) - 1st, 2nd fc layers

learning rate : 초기값 - $10^{-2}$ , validation accuracy 증가가 멈출 때마다 10배씩 줄임

→ 3 times 감소하게 됨, 370K(74 epochs) 반복해서 멈춤

논문의 저자들은 많은 파라미터와 깊은 깊이를 가진 nets가 ‘implicit regularisation’와 특정층의 레이어 초기값으로 인해서 적은 pochs로 수렴된다 짐작했다. The initialisation of the network weights이 중요하다 판단함.

→ 문제점 : 깊은 nets에서 나쁜 초기값은 학습을 멈추게 하거나 gradient를 불안정하게 할 수 있다.

→ 해결 방법 : Table 1의 상대적으로 적은 레이어인 A configuration 으로 랜덤 초기값으로 훈련 반복(learning rate 감소안함) → training architectures가 더 깊어졌을 때 해당 값으로 처음 4개 conv layers와 마지막 fc layers 3개 초기값 설정(나머지는 랜덤 , 범위는 0 ~ $10^{-2}$), biases : 0로 초기화. 

→ 결론 : 타 논문을 통해 pre-training 없이도 초기값 설정이 가능함을 알아냄.

고정된 이미지 224x224 (하지만 다양하게 하기 위해서)

랜덤으로 이미지를 rescale하여 224 크기로 자름

자른 이미지를 수평으로 뒤집거나 랜덤하게 RGB 값을 변경

384 크기의 이미지로 사전훈련된 a single-scale model 의 fine-tuning을 이용해서 같은 configuration으로 multi-scale models를 훈련했다.

1. Testing

test input scale과 train input scale이 같지 않아도 된다.

horizontal flipping한 이미지도 testing에 사용했다.

앞서 3개의 fc layer를 test 할 때 아래와 같이 바꿈 = the fully-convolutional network

1st fc layer → 7x7 conv. layer

2nd, 3rd fc layer → 1x1 conv. layer

자르지 않은 전체 이미지가 적용됨(conv.로만 이루어졌기 때문)

따로 이미지를 잘라서 input값으로 넣는 작업을 안해도 된다

마지막 레이어 결과 = a class score map size = 이미지 크기가 같으면 늘 같은 값이 나옴 → 해당 값이 softmax 에 넘어감

final score = spatially averaged = original images와 flipped images 의 평균값

- ?
    
    The result is a class score map with the number of
    channels equal to the number of classes, and a variable spatial resolution, dependent on the input
    image size. Finally, to obtain a fixed-size vector of class scores for the image, the class score map is
    spatially averaged (sum-pooled). We also augment the test set by horizontal flipping of the images;
    the soft-max class posteriors of the original and flipped images are averaged to obtain the final scores
    for the image.
    

multi-crop evaluation 

서로 다른 conv. 경계 조건을 가져서 dense evaluation과 상호보완적인 것

ConvNet 을 a crop에 적용 → paddig : 0

receptive field 상당히 증가

dense evaluation

0 padding이 아닌 crop 한 주변의 부분들을 padding 값으로 사용

receptive field 크기 증가

1. Implementation details

### Classification experiments

1. Single scale evaluation

scale 값을 두 가지(256, 384) 고정하고 훈련 

256x256 : 사전 훈련용, 이후 384 훈련에 사전 훈련 모델로 사용됨, lr: $10^{-3}$

384x384 : 256 사전 훈련을 사용하기 때문에 속도가 더 빠름, 256 훈련시 나온 가중치 사용, 더 작은 learning rate 값 사용

1. Multi-scale evaluation

S 값을 multi-scale 을 통해서 설정한다

256 ~ 512 사이의 S로 랜덤 샘플링하여 훈련 (=scale jittering)

속도의 이유로 앞서 384로 훈련한 동일 configuration의 single scale model의 모든 레이어를  fine-tuning하여 multi-scale models를 훈련

1. Multi-crop evaluation
2. ConvNet fusion

1. Comparison with the state of the art

7.3% 에러로 2등을 했다. 앙상블 2개 모델을 이용한 결과로는 6.8% 에러율이 나왔다.

### Conclusion

이미지 분류를 위한 최대 19개 층의 convolutional networks를 만들었다. 큰 데이터셋임에도 convolutional networks로 깊은 레이어 설계가 가능한 것을 보여줬고, 깊이가 중요성을 알려준다.