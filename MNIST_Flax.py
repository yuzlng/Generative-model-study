##### 2. Load the MNIST dataset #####

import tensorflow_datasets as tfds  
import tensorflow as tf  

tf.random.set_seed(0)  

train_steps = 1200  # 학습 과정에서의 총 step 수
eval_every = 200  # 몇 step마다 모델을 평가할지
batch_size = 32 # 한 번에 학습에 사용할 sample의 개수

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

# MNIST 데이터셋 : 이미지를 픽셀값(0~255)으로 표현 -> 픽셀값을 0~1 범위로 정규화
train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  

train_ds = train_ds.repeat().shuffle(1024)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)




##### 3. Define the model with FLAX NNX #####
from flax import nnx
from functools import partial

class CNN(nnx.Module):  # 합성곱 신경망 : 작은 크기의 필터(커널)을 사용하여 이미지의 패턴을 학습

  def __init__(self, *, rngs: nnx.Rngs):   
    self.conv1 = nnx.Conv(1,32,kernel_size=(3, 3), rngs=rngs) #입력 이미지를 32개의 특징 맵으로 변환
    self.conv2 = nnx.Conv(32,64,kernel_size=(3, 3), rngs=rngs) #첫 번째 레이어의 출력을 64개의 특징 맵으로 변환
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs) # 첫 번째 레이어 : 3136개의 입력을 받아 256개의 중간 출력 생성
    self.linear2 = nnx.Linear(256, 10, rngs=rngs) # 두 번째 레이어 : 256개의 중간 출력을 받아 10개의 클래스(0~9)로 분류

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))  #nnx.relu : ReLU 활성화 함수, 입력값의 음수를 0으로 변환하고 양수는 그대로 반환 -> 신경망에 비선형성을 추가 
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten, 입력 : (32, 5, 5, 64)-> 출력 : (32, 1600) (5*5*64=1600)
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x  # (32, 10)

# Instantiate the model.
model = CNN(rngs=nnx.Rngs(0))
# Visualize 
# nnx.display(model)

# Run the model 
# import jax.numpy as jnp  
# y = model(jnp.ones((1, 28, 28, 1)))
# print(y)




##### 4. Create the optimizer and define some metrics #####
import optax # JAX 기반 최적화 라이브러리 -> 여기선 adamw를 쓸 것

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(  #성능지표 정의 
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

# Optimizer의 설정 및 상태를 확인
nnx.display(optimizer)




##### 5. Define training step functions #####