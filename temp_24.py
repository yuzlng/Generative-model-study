##### 2. Load the MNIST dataset ##### -- PyTorch data loader 
train_steps = 1200  # 학습 과정에서의 총 step 수
eval_every = 200  # 몇 step마다 모델을 평가할지
batch_size = 32 # 한 번에 학습에 사용할 sample의 개수

import numpy as np
import jax.numpy as jnp 
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader): # 각각 뭔지 찾아보기 
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0, # CPU 
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

# Train 및 Test 데이터셋 로드
mnist_dataset = MNIST('/tmp/mnist/', train=True, download=True, transform=FlattenAndCast())
test_dataset = MNIST('/tmp/mnist/', train=False, download=True, transform=FlattenAndCast())

# DataLoader 정의
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_ds = NumpyLoader(test_dataset, batch_size=batch_size, num_workers=0)




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
# nnx.display(optimizer)




##### 5. Define training step functions #####
def loss_fn(model: CNN, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()  # 예측값(logits)과 실제 레이블(labels) 간의 교차 엔트로피 손실을 계산
    return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric,batch):
   """Train for a single step."""
   grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)  # loss_fn의 출력값과 기울기를 동시에 계산, has_aux=True : 손실 외의 추가 출력값(로짓)을 반환
   (loss, logits), grads = grad_fn(model, batch)
   metrics.update(loss=loss, logits=logits, labels=batch['label'])  # 성능 지표(metrics)를 업데이트
   optimizer.update(grads)  # model parameter 업데이트 

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label']) # In-place updates




##### 6. Train and evaluate the model #####
from IPython.display import clear_output
import matplotlib.pyplot as plt

metrics_history = { # 성능 기록용 딕셔너리
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

for step, batch in enumerate(training_generator): # training generator로 사용 (Pytorch 버전)
  # Run the optimization for one step and make a stateful update to the following:
  # - The train state's model parameters
  # - The optimizer state
  # - The training loss and accuracy batch metrics
  images, labels = batch  # 튜플 언패킹
  batch = {'image': images.reshape(-1, 28, 28, 1), 'label': labels}  # 딕셔너리로 변환
  
  train_step(model, optimizer, metrics, batch)

  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # one epoch 마다 
    # Log the training metrics.
    for metric, value in metrics.compute().items():  
      metrics_history[f'train_{metric}'].append(value)  
    metrics.reset()  

    # Compute the metrics on the test set after each training epoch.
    for test_batch in iter(test_ds):
      images, labels = test_batch
      test_batch = {'image': images.reshape(-1, 28, 28, 1), 'label': labels}
      eval_step(model, metrics, test_batch)


    # Log the test metrics.
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()  
    

    # 결과 시각화
    clear_output(wait=True)    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    for dataset in ('train', 'test'):
      ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
      ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
    ax1.legend()
    ax2.legend()
    plt.show()





##### 7. Perform inference on the test set #####
model.eval() # evalution mode로 switch

@nnx.jit
def pred_step(model: CNN, batch):
  logits = model(batch['image'])
  return logits.argmax(axis=1)

test_batch = next(iter(test_ds))
images, labels = test_batch
test_batch = {'image': images.reshape(-1, 28, 28, 1), 'label': labels}

pred = pred_step(model, test_batch)


fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
  ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
  ax.set_title(f'label={pred[i]}')
  ax.axis('off')
