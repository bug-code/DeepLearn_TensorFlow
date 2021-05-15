import tensorflow as tf
import tensorflow.keras as keras
from  tensorflow.keras import layers , models , Sequential , losses , optimizers , datasets

'''
自定义网络层继承自layers.Layer基类 
自定义网络类继承自keras.Model基类
'''

'''
自定义网络层：
    至少需要实现
                1、初始化__init__方法
                2、前向传播逻辑call方法
'''
#自定义网络层继承layers.Layer基类
class MyDense(layers.Layer):
    #实现子类初始化函数
    def __init__(self , in_dim , out_dim):
        #调用基类初始化函数
        super(MyDense , self).__init__()
        #根据输入输出，初始化网络层权值，并将该张量设置为需要优化
        self.kernel = self.add_variable('w' , [in_dim , out_dim] , trainable = True)
        #self.add_variable返回张量w的python引用

    #实现前向传播逻辑
    def call(self , inputs  ,training = None):
        #实现自定义网络层前向计算逻辑
        out = inputs@self.kernel
        #将激活函数设置为relu函数，此处可以自定义
        out = tf.nn.relu(out)
        return out

'''
使用自定义网络层封装成一个网络模型
net = Sequential([
    MyDense(784 , 256) , 
    MyDense(256 , 128) , 
    MyDense(128 , 64)  ,
    MyDense(64 , 32)   ,
    MyDense(32 , 10)   ,   
])
net.build(input_shape=(None , 28*28))
net.summary()
'''

'''
自定义网络模型类：
                1、创建自定义网络模型类 ， 继承自Model基类
                2、 创建对应的网络层对象 
该方法更加自由和灵活，当需要创建自己定义的网络模型架构时使用。
'''

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel , self).__init__()
        self.fc1 = MyDense(28*28 , 256)
        self.fc2 = MyDense(256 , 128)
        self.fc3 = MyDense(128 , 64)
        self.fc4 = MyDense(64 , 32)
        self.fc5 = MyDense(32 , 10)
    def call(self , inputs , training = None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
        


