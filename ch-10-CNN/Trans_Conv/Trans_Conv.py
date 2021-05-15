import tensorflow as tf
import tensorflow.keras as keras

'''
转置卷积：指卷积核矩阵W产生的稀疏矩阵W'在计算过程中需要先转置W'T，
再进行矩阵相乘运算，而普通卷积并没有转置W'的步骤。

转置卷积：通过在  输入  之间填充大量的padding来实现输出高宽大于输入高宽的效果，从而实现向上采样的目的。
其本质就是卷积输入的填充，书中10.2.5节，其中转置的含义不是数学上矩阵转置的含义，而应当是输出与普通卷积相同的大小。
例如：普通卷积输入大小a , 输出大小b , 
    转置卷积输入大小b , 输出大小a , 
转置卷积的实现过程
'''


'''
转置卷积原理过程：


先计算普通卷积
'''
#tf.nn.conv2d基于输出X:[b,h,w,c_in]和卷积核W：[k ,k,c_in , c_out]进行卷积运算
x = tf.range(25)+1
#表示张量batch=1 , hight=5 ， weight = 5 ， channel = 1 表示1个5x5大小1通道的图片 
x = tf.reshape(x , [1,5,5,1])
x = tf.cast(x , tf.float32)
# print(x)

#创建一个3x3的卷积核 ， 形式为[3,3,1,1] , 即大小为3x3 , 输入通道为1，输出通道为1
w = tf.constant([[-1,2,-3],[4,-5,6],[-7,8,-9]])
#添加输入通道维度
w = tf.expand_dims(w , axis=2)
#添加输出通道维度
w = tf.expand_dims(w , axis=3)
w = tf.cast(w , tf.float32)  
# print(w)

#普通卷积输出,不填充
out = tf.nn.conv2d(x , w , strides=2 , padding='VALID')
print('普通卷积输出',out , '\n')
'''
将普通卷积结果作为转置卷积运算的输入  ， ConvTranspose(out)
'''
trans_out = tf.nn.conv2d_transpose(out , w , strides = 2 
, padding = 'VALID' , output_shape=[1,5,5,1])
print('转置卷积输出',trans_out , '\n')

'''不整除情况'''
#卷积层卷积时，输出向下取整，这意味着后面的小于k的部分不会参与卷积运算
x1 = tf.random.normal([1,6,6,1])
out1 = tf.nn.conv2d(x1 , w , strides=2 , padding='VALID')
print('普通卷积不整除输出' ,out1)


'''
转置卷积实现
转置卷积核的定义格式为:[k,k,c_out , c_in]与普通卷积核定义[k,k,c_in,c_out]不同
'''

x2 = tf.range(16)+1
x2 = tf.reshape(x2 , [1,4,4,1])
x2 = tf.cast(x2 , tf.float32)


w1 = tf.constant([ [-1,2,-3] , [4,-5,6] , [-7,8,-9] ])
w1 = tf.cast(w1 , tf.float32)
w1 =tf.expand_dims(w1 , axis=2)
w1 = tf.expand_dims(w1 , axis=3)
out3 = tf.nn.conv2d(x2 , w1 , strides =1 , padding ='VALID' )
# out3 = tf.squeeze(out3)
# print(out3)

#恢复与输入同大小高宽的张量
#tf.nn.conv2d_transpose进行转置卷积运算时，
# 1、需要额外手动设置输出的高宽
# 2、padding只能设置为VALID或者SAME
out3 = tf.nn.conv2d_transpose(out3,w1,strides=1,padding='VALID',output_shape=[1,4,4,1])
out3 = tf.squeeze(out3)

print(out3)