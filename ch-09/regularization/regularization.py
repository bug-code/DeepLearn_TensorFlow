import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import  Sequential , Model  , losses , metrics , optimizers , regularizers
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
N_epochs = 30
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


'''
实验正则化系数λ对神经网络模型的影响，使用L2正则化，构建5层神经网络(包含输入层和输出层)
2，3，4层添加L1正则化约束
'''

#构建带正则化优化项的神经网络
def buid_model_with_regularzation(_lambda):
    model = Sequential()
    model.add(Dense(8,input_dim = 2 , activation='relu'))
    for i in range(3):
        model.add(Dense(256 , activation='relu' , kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy' , optimizer='adam'  , metrics=['accuracy'])
    return model


#5个实验
for _lambda in [0.00001 , 0.0001 , 0.001 , 0.01  , 0.1]:
    model = buid_model_with_regularzation(_lambda)
    history = model.fit(x_train ,  y_train , batch_size=200,epochs=N_epochs , verbose=1)
    title ='正则化-[λ={}]'.format(str(_lambda))
    filename = '正则化_%f.png'%(_lambda)
    #设置横纵坐标范围，在其内进行模型采样
    xx = np.arange(-2,3,0.01)
    yy = np.arange(-2,2 ,0.01)
    #采样
    XX  , YY= np.meshgrid(xx , yy)
    preds = model.predict_classes(np.c_[XX.ravel() , YY.ravel()])
    draw(X , Y , title , filename , XX , YY , preds) 
