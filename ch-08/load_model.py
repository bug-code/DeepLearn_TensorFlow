import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers , Sequential

'''
常用网络模型无需手动创建，直接从keras.applications子模块中创建并使用。
可通过设置weights参数加载预训练的网络参数。

例如:
    1、将ResNet50去除最后一层后的网络作为新任务的特征提取子网络。
    2、设置weights为要测试的数据集，对网络参数初始化
    3、根据自定义任务的类别最佳一个对应数据类别数的全连接分类层或子网络
可以在预训练网络的基础上快速、高效地学习新任务
'''

#ResNet50只保存了对于imageNet数据集地预训练模型参数。如果需要测试其他数据集，需要提前训练好并保存
#设置weights参数，为ResNet50网络模型设置参数。include_top=False 去除resnet网络参数的最后一层
#resnet子网络作为特征提取子网络
resnet = keras.applications.ResNet50(weights='imagenet' , include_top=False )
#冻结自网络参数，不参与训练
resnet.trainable=False
'''
子网络测试
'''
# resnet.summary()
# x = tf.random.normal([20 , 224, 224 , 3])
# out = resnet(x)
# print(out.shape)

#在resnet特征提取网络后添加一个池化层进行姜维
#创建池化层函数 layers.GlobalAveragePooling2D()
global_average_layer = layers.GlobalAveragePooling2D()
#测试池化层输出测试
# out = global_average_layer(out)
# print(out.shape)


#新建全连接层，设置输出节点为100
#该全连接层的输出为100个类别的概率分布

#新建全连接层
last_fc = layers.Dense(100)
#最后全连接层测试
# out = last_fc(out)
# print(out[:10] , '\n','------------------------------------')
# print(out.shape)

#将以上网络封装成一个新网络模型
mynet = Sequential([resnet , global_average_layer , last_fc ] )
mynet.summary()