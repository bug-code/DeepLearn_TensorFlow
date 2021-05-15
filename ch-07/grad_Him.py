import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
注意区分梯度下降和反向传播算法

反向传播算法：反向传播是在神经网络算法中更新权值参数的方法，而不是寻找最小值的方法
反向传播是为了更新神经网络权值参数，使得神经网络拥有更好的性能。
反向传播算法与梯度下降算法会混淆原因在于：反向传播算法是基于梯度下降算法寻找到的最小值
来更新神经网络参数的。因此如果需要，可以使用各种最优化算法来代替梯度下降算法

梯度下降的含义是指：函数沿着某个方向所能找到的最小值。该最小值为全局最小值，但不一定是全局最小值。
在算法的执行效率，算法不仅与实现算法的方式有关，也与算法的初始化有关。不同的数值初始化也会影响
函数寻找最小（最大）值的方向。也就是说如果初始化不合适，可能会导致找不到最值。
调参侠的原因
'''
def himmelblau(x ):
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
def draw( ):
    x = np.arange(-6 , 6 , 0.1)
    y = np.arange(-6 , 6 , 0.1)
    #采样
    X , Y =np.meshgrid(x,y)
    z = himmelblau([X , Y])
    
    #画图
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X , Y  , z)
    ax.view_init(60,-30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def find_min(x , epoch ,  lr):
    x = tf.constant(x)
    for step in range(epoch):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)
        #计算各元导数
        grads = tape.gradient(y , [x])[0]
        x -= lr*grads
        if step%20==19:
            print('step {}: x = {} , f(x) = {} '.format(step,x.numpy(),y.numpy()))


if __name__ == "__main__":
    draw()
    find_min([4. ,  0.] , 200 , 0.01)