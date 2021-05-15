import tensorflow as tf
'''
tf.conv2d函数可以方便地实现2D卷积运算，
输入X:[b,h,w,Cin]
卷积核：[k,k,Cin,Cout]
输出：[b , h' , w' , Cout]
Cin表示输入通道数




Cout表示卷积核数量
'''

#模拟输入，3通道，高5
x = tf.random.normal([2,5,5,3])
#创建W张量，4个3x3大小地卷积核
w = tf.random.normal([3,3,3,4])
#设置偏置
b = tf.zeros([4])
#设置卷积层的步长和填充
'''
填充格式：
    padding=[[0,0] , [上 ， 下] , [左 ， 右] ， [0,0]]
    通过设置参数padding='SAME' ， strudes = 1可以直接得到输入、输出同大小的卷积层
    具体填充数量由tensorflow自动计算完成
'''

out = tf.nn.conv2d(x , w , strides=1 , padding=[[0,0],[0,0],[0,0],[0,0]])
#添加偏置
out += b
print(out)



