from time import perf_counter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

N = 100
x = tf.random.uniform((N, N), dtype=tf.dtypes.float64)
t0 = perf_counter()
s, u, v = tf.linalg.svd(x)
elapsed_time = perf_counter() - t0
print("Execution time: ", elapsed_time)
print("Result: ", tf.reduce_sum(s).numpy())
