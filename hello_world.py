import numpy as np
import pickle
import os

s = os.getcwd()
print("cwd is: ", s)

A = np.arange(16).reshape(4, 4)
print(A)

with open("a.pkl", "rb") as f:
    A = pickle.load(f)
print("New A is: ")
print(A)
print("Hello world")
print("abcdefg")
print("a IS: ", A)
print("success")

print("auto sync does not always work")
print("Use ssh in the remote interpreter config!")

print("finally")

import tensorflow as tf
m = tf.constant([5])
with tf.Session() as sess:
    mValue = sess.run(m)
print("m value is:", mValue)