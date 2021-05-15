import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , optimizers , Sequential , losses , metrics
'''
卷积层的使用
'''
#模拟输入
x = tf.random.normal([2,5,5,3])

#创建4个3x3大小的卷积核的卷积层 ，步长为1 ， padding方案为'SAME'
layer = layers.Conv2D(4 , kernel_size=3 , strides=1 , padding='SAME')
#通过trainable_variables返回w和b

#需要调用__call__函数进行初始化
out = layer(x)
#返回所有需优化的张量
vars = layer.trainable_variables

#使用layer.kernel  、 layer.bias返回张量w和b
w = layer.kernel
b = layer.bias

print(out,'\n' , '--------------------------------------------')
print(vars,'\n' , '--------------------------------------------')
print(w,'\n' , '--------------------------------------------')
print(b,'\n' , '--------------------------------------------')
