import tensorflow_datasets as tfds  
import tensorflow as tf  

tf.random.set_seed(0)  

train_steps = 1200
eval_every = 200
batch_size = 32