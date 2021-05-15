import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , optimizers , Sequential , losses , metrics

'''
ResNet通过在卷积层的输入和输出之间添加Skip Connection
实现层数回退机制

深度残差网络特点：通过堆叠残差模块，达到了较深的网络层数，
从而获得训练稳定、性能优越的深层网络模型
'''

#ResBlock实现

#创建一个新类，在初始化阶段创建残差块中需要的卷积层，激活函数等

#创建F(x)卷积层
class BasicBlock(layers.Layer):
    #残差模块类
    def __init__(self , filter_num , stride = 1):
        super(BasicBlock , self).__init__()
        #f(x)包含了2个普通卷积层
        #创建卷积层1
        self.conv1 = layers.Conv2D(filter_num , kernel_size=3,strides=stride , padding='SAME')
        #参数标准化层
        self.bn1 = layers.BatchNormalization( )
        #激活函数层
        self.relu = layers.Activation('relu')
        #创建卷积层2
        self.conv2 = layers.Conv2D(filter_num , kernel_size=3 ,strides=1 , padding='SAME')
        self.bn2 = layers.BatchNormalization()
        #identity(x)卷积层，当F(x)的形状与x不相同时，需要进行形状转换
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsample=lambda x: x
    
    def call(self , inputs , training=None):
        print(inputs)
        out = self.conv1(inputs)
        print('shape:' ,out , '\n')
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        print('shape:' ,out , '\n')
        out = self.bn2(out)
        identity = self.downsample(inputs)
        print('shape:' ,out , '\n')
        output = layers.add([out , identity])
        output = tf.nn.relu(output)
        return output

#basicBlocl网络模型测试

#模拟输入
x = tf.range(25)+1
#表示张量batch=1 , hight=5 ， weight = 5 ， channel = 1 表示1个5x5大小1通道的图片 
x = tf.reshape(x , [1,5,5,1])
x = tf.cast(x , tf.float32)

#创建basicBlock对象

#在进行填充时，如果padding='SAME',只有当strides=1，输入与输出才会大小相同
basicblock = BasicBlock(1,stride=1)
output = basicblock.call(x)
print(output)



