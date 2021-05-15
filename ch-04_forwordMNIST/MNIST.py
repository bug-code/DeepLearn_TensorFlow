import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers ,optimizers,datasets,metrics

def data_init():    
    #数据预处理
    (x_train,y_train) , (x_test,y_test) = datasets.mnist.load_data()

    x_train=2*tf.convert_to_tensor(x_train,dtype=tf.float32)/255 - 1
    x_test=2*tf.convert_to_tensor(x_test,dtype=tf.float32)/255 - 1

    y_train=tf.convert_to_tensor(y_train,dtype=tf.int32)
    y_test=tf.convert_to_tensor(y_test,dtype=tf.int32)

    '''手写数字集用作神经网络训练数据集进行分类，由于是多分类，将分类结果分成10种，将分类结果用一个一维数组表示：即为onehot编码，
    例如分类结果为6，onehot编码结果【0，0，0，0，0，0，1，0，0，0】
    '''
    y_train=tf.one_hot(y_train,depth=10,on_value=None,
    off_value=None)
    y_test=tf.one_hot(y_test,depth=10,on_value=None,
    off_value=None)
    # print(y_train.shape[0] , y_train.shape[0])
    train_dataset =tf.data.Dataset.from_tensor_slices((x_train,y_train))
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))

    ##将训练数据分为200片，每片300张图片进行并行训练
    ##每片执行一次后再执行一次
    train_dataset=train_dataset.batch(200)
    test_dataset=test_dataset.batch(200)

    ##设置学习率，随机梯度下降
    optimizer=optimizers.SGD(learning_rate=0.001)
    #建立测量尺，计算精度函数
    acc_meter=metrics.Accuracy()
    return train_dataset ,x_train, x_test , y_test , optimizer , acc_meter

def create_net( ): 
    ###搭建网络模型
    '''
    tensorflow中的Sequential容器搭建多层网络
    '''
    model=keras.Sequential([
        #ReLU函数有良好的非线性特定，梯度计算简单，训练稳定，
        # 是深度学习模型使用最广泛的激活函数之一

        #第一层网络256个输出
        layers.Dense(256,activation='relu'),
        #第二层网络128个输出  
        layers.Dense(128,activation='relu'),
        #输出层10个，最后分类结果也是10类
        layers.Dense(10)
    ])

    return model



'''
一个epoch就是将所有数据训练一遍
enumerate将可遍历对象组合成一个索引序列
'''

#模型训练

def train_epoch(epoch,train_dataset , model , optimizer):
    for step,(x,y) in enumerate(train_dataset):
        
        #记录下所有可训练变量的梯度，以便下一步更新
        with tf.GradientTape() as tape:

            x=tf.reshape(x,(-1,28*28))
            out=model(x)
            #计算单个方差
            loss=tf.square(out-y)
           
            #张量求和，也就是计算所有图片的均方误差和，计算均方差
            loss=tf.reduce_mean(loss)
            print(loss)
        #获取每个变量对应的梯度值
        grads=tape.gradient(loss,model.trainable_variables)
        #根据每个变量的梯度值，更新相关变量的模型参数
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        #显示每训练100张图片之后的loss值
        if (step%100==0):
            print(epoch,step,'loss:',loss)
    return model

def quiz_epoch(x_test,y_test , model , acc_meter):
    x_test=tf.reshape(x_test,(-1,28*28))
    out = model(x_test)
    #计算每一个epoch后的模型准确度
    acc_meter.update_state(out,y_test)
    print("acc:",acc_meter.result().numpy())

    return acc_meter.result().numpy()



    
    


if __name__ == '__main__':
    train_dataset , x_train,x_test , y_test , optimizer , acc_meter = data_init()
    model=create_net()
    for i in range(20):
        model=train_epoch( i,train_dataset , model , optimizer)
        acc=quiz_epoch(x_test,y_test , model , acc_meter)