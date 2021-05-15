import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x = tf.constant([[[1.,2.,3.],[0.,2.,4.],[4.,3.,2.] ],[ [1.,2.,3.],[0.,2.,4.],[4.,3.,2.]]])
layer = layers.Softmax(axis=2)
out = layer(x)
print(out) 