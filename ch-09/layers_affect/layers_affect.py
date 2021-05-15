import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import  Sequential , Model  , losses , metrics , optimizers
from tensorflow.keras.layers import Dense  
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import  numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''
构建数据集：
    月牙形状数据集，初始化处理，数据集样本特征向量长度为2 ，类别为2，类别标签为0/1

    采样1000个样本数据，同时添加标准差为0.25的高斯噪声数据
'''
#从moon分布中随机采样1000个样本点，切分为训练集-测试集
N_SAMPLES = 1000
TEST_SIZE = 0.3
X  , Y =make_moons(n_samples=N_SAMPLES , noise=0.25 , random_state=100)
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = TEST_SIZE , random_state = 42)
#绘制数据集分布图
def draw(X , Y , title , filename , XX = None , YY=None , preds = None):
    plt.figure(figsize=(16,10))
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title(title)
    plt.scatter(X[:,0] , X[:,1] , c=Y.ravel() , s=40  ,cmap=plt.cm.Spectral, edgecolors='none')
    
    #根据网络输出绘制预测曲面图
    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX , YY , preds.reshape(XX.shape) , 25  , alpha = 0.08 , camp = plt.cm.Spectral)
        plt.contour(XX , YY , preds.reshape(XX.shape) , levels = [.5] , cmap="Greys" , vmin=0 , vmax = .6)

    plt.savefig(filename)
    plt.show()

# draw(X , Y , 'dataset distribute' , 'dataset')
'''
实验网络层数对于数据的过拟合程度。进行5次实验，
在n∈[0,4]时， 构建网络层数为n+2层的全连接层网络，（因为还有输入层和输出层）
通过adam优化器训练500个epoch，获得分隔曲线图
'''

for n in range(5):
    #创建容器
    model = Sequential()
    #创建第一层
    model.add(Dense(3 , input_dim=2 ,activation='relu'))
    for i in range(n):
        model.add(Dense(32  ,activation='relu'))
    #添加输出层
    model.add(Dense(1 ,activation='sigmoid'))
    model.compile(loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])
    history = model.fit(x_train , y_train , batch_size=100 , epochs=100 , verbose=1)
    #设置横纵坐标范围，在其内进行模型采样
    xx = np.arange(-2,3,0.01)
    yy = np.arange(-2,2 ,0.01)
    #采样
    XX  , YY= np.meshgrid(xx , yy)
    preds = model.predict_classes(np.c_[XX.ravel() , YY.ravel()])
    title = '网络层数（{}）'.format(n)
    filename = '网络容量%f.png'%(2+n*1)
    draw(X , Y , title , filename , XX , YY , preds)
        