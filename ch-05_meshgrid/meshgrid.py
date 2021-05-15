import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''通过tf.meshgrid函数方便生成二维网格的采样点坐标，方便可视化等应用场景
'''
#设置x轴采样点
x = tf.linspace(-8,8,100)
#设置y轴采样点
y = tf.linspace(-8,8,100)

#生成网格点，内部拆分后返回
#tf.meshgrid会返回在axis=2维度切割后的2个张量A和B，其中张量A包含了所有点的X坐标，B包含了所有点y坐标
#shape都为[100,100]
x , y = tf.meshgrid(x,y)
z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z

#绘制3D图像
fig = plt.figure()
ax = Axes3D(fig)

ax.contour3D(x.numpy(),y.numpy(),z.numpy(),3000)
plt.show()