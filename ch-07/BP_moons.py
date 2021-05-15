# import tensorflow as tf
from sklearn.datasets import make_moons 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
'''
手工实现反向传播算法。数据集 make_moons 两个属性，两个类别
神经网络四层：输入2节点 ， 隐藏层1：25 隐藏层2：50 隐藏层3：25 输出层：2 ， 激活函数sigmoid
'''
def draw_mse_acc(epoch_sumloss , epoch_acc):
    x=[i for i in range(len(epoch_sumloss))]
    #左纵坐标
    fig , ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss' , color=color)
    ax1.plot(x , epoch_sumloss , color=color)
    ax1.tick_params(axis='y', labelcolor= color)

    ax2=ax1.twinx()
    color1='blue'
    ax2.set_ylabel('acc',color=color1)
    ax2.plot(x , epoch_acc , color=color1)
    ax2.tick_params(axis='y' , labelcolor=color1)

    fig.tight_layout()
    plt.show()


def draw(X , Y):
    plt.figure(figsize=(16,10))
    plt.scatter(X[:,0] , X[:,1] , c=Y.ravel() , s=40  ,cmap=plt.cm.Spectral, edgecolors='none')
    plt.show()

'''
数据初始化
'''
def  data_init(N_SAMPLES, TEST_SIZE):
    #使用make_moons函数生成数据集
    X , Y = make_moons(n_samples=N_SAMPLES  ,noise=0.2 , random_state=100)
    #切分数据集
    x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = TEST_SIZE , random_state=42)

    

    
    return X , Y , x_train , y_train , x_test , y_test




'''
单个网络层实现
'''
class Layers(object):
    def __init__(self,n_input , n_neurons , activation=None , weights = None , bias = None):
        '''
        n_input表示输入节点数
        n_neurons表示输出节点数
        activation表示激活函数
        weights表示神经网络参数权重矩阵
        bias表示偏置
        '''
        #如果未设置权重，则正态初始化权重W，同时将其压缩在[-1,1]之间
        self.weights = weights if weights is not None else np.random.randn(n_input , n_neurons) * np.sqrt(1/n_neurons) 
        self.bias = bias if bias is not None else np.random.rand(n_neurons)*0.1
        self.activation = activation
        self.out_activation = None
        self.error  = None
        self.delta = None

    #激活函数
    #根据初始化的激活函数，将计算的结果经过选择的激活函数处理
    def _apply_activation(self,result):
        if self.activation is  None:
            result = result
        elif self.activation == 'relu':
            result =  np.maximum(result , 0)
        elif self.activation == 'tanh':
            result =  np.tanh(result)
        elif self.activation == 'sigmoid':
            result =  1/(1+np.exp(-result))
        
        return result

    #前向传播
    def activate(self , x ):
        
        result  = np.dot(x,self.weights ) +self.bias
        #获得该层网络输出
        self.out_activation=self._apply_activation(result)
        return self.out_activation

    #计算从激活函数输出到激活函数输入的导数
    def apply_activation_derivative(self , out):
        if self.activation is None:
            #如果未使用激活函数，则其导数为1。使用ones_like函数返回一个与结果形状一样的矩阵
            result =  np.ones_like(out)
        elif self.activation == 'relu':
            #将激活函数输出的张量转换为array数组，并复制
            grad = np.array(out , copy=True)
            #根据relu激活函数的定义，计算激活函数的导数
            grad[out > 0] = 1
            grad[out <= 0] = 0
            result = grad
        elif self.activation == 'tanh':
            result = 1 - out**2
        elif self.activation == 'sigmoid':
            result =  out*(1-out)
        
        return result

class net(object):
    def __init__(self):
        self._layers=[]
    
    def add_layer(self , layer):
        self._layers.append(layer)

    #网络模型前向计算
    def feed_forward(self , x):
        for layer in self._layers:    
            x = layer.activate(x)  
        return x
        
    #反向传播算法
    def backpropagation(self , X , Y , lr ):
        out = self.feed_forward(X)
        #反向循环
        for i in reversed(range(len(self._layers))):
            #获得当前层对象
            layer = self._layers[i]
            #输出层和隐藏层的梯度计算公式不一样
            if layer == self._layers[-1]:
                layer.error = Y - out
                layer.delta = layer.error * layer.apply_activation_derivative(out)
            else:
                next_layer = self._layers[i+1]
                '''
                隐藏层误差：是与最终输出预测相关的，所以从后往前计算时，
                使用后面一层网络的Δ乘上后面一层网络的权重参数。
                Δ表示当下输出在该偏导方向与实际值的差距。
                '''
                layer.error = np.dot(next_layer.weights , next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.out_activation)

        #网络参数更新
        #网络当下权重-网络权重Δ*该层网络的输入*学习率
        for i in range(len(self._layers)):
            layer = self._layers[i]
            #如果该网络层为输入层，则该网络层的输入即为原始数据，否则为上一层网络的输出
            o_i = np.atleast_2d(X  if i == 0 else self._layers[i-1].out_activation)
            layer.weights += layer.delta * o_i.T * lr

    
    #计算准确率
    def accuracy(self , predict , y):
        
        #最简单计算准确率方法
        '''right_sum = 0
        for i in range(len(predict)):
            if predict[i]==y[i]:
                right_sum +=1
        acc = right_sum / len(predict)
        '''
        #书上的方法
        '''return np.sum(np.equal(np.argmax(predict, axis=1), y)) / y.shape[0]'''
        #我的方法
        acc = ( len(predict) -sum((predict^y)) ) / len(predict)
        return acc

        
    #预测
    def predict(self , dataset):
        pre_result= []
        for i in range(len(dataset)):
            pre = self.feed_forward(dataset[i])
            if pre[0] >=pre[1]:
                re = 0
            else:
                re = 1
            pre_result.append(re)
        # return self.feed_forward(dataset)
        return pre_result


    #训练
    def train(self , x_train , y_train , x_test , y_test , lr , epoch):
        # 训练集进行onehot编码，这一步很精妙。
        #由于只有两个类别，构建一个 行数为：训练集标签数 ， 列数为：2的二维数组并以0进行填充
        y_onehot = np.zeros((y_train.shape[0] , 2))
        #由于原始数据标签是一个一维数组，使用np.arange构建一个[0,1,..,训练集标签数]一维数组
        #以该数组作为y_onehot的横坐标，y_train也是一个一维以0和1组成的数组作为纵坐标
        #对y_onehot进行赋值即可完成onehot编码
        y_onehot[ np.arange(y_train.shape[0]) , y_train.astype('int64')] = 1

        # y_onehot = y_train.astype('int64')
        mses = []
        accs = []
        for i in range(epoch):
            for j in range(len(x_train)):               
                self.backpropagation(x_train[j] , y_onehot[j] , lr)
            #每计算10个数据，记录一次误差
            if i%10==0:
                mse = np.mean(np.square(y_onehot-self.feed_forward(x_train)))
                acc = self.accuracy(self.predict(x_test),y_test.flatten())*100
                mses.append(mse)
                accs.append(acc)
                print('epoch:%s , MSE:%f'%(i , float(mse)))
                print('accuracy : %.2f%%'% acc)
        return mses , accs
                
if __name__ == "__main__":
    X , Y , x_train , y_train , x_test , y_test = data_init(2000 , 0.3)
    draw(X , Y)
    layer1 = Layers(2,25,'sigmoid')
    layer2 = Layers(25,50 ,'sigmoid')
    layer3 = Layers(50 ,25 ,'sigmoid')
    layer4 = Layers(25 ,2 ,'sigmoid')
    nn = net()
    nn.add_layer(layer1)
    nn.add_layer(layer2)
    nn.add_layer(layer3)
    nn.add_layer(layer4)
    mse , acc = nn.train(x_train , y_train , x_test , y_test , 0.01 ,1000)
    draw_mse_acc(mse , acc)



