import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as datasets

#数据初始化
'''
初始化时，需将数据集分为训练集和测试集

1.数据归一化
2.数据转换为张量
3.需要将y进行ont-hot编码
4.数据聚合并切片
'''
def _init( ):
    #load数据集
    (x_train,y_train) , (x_test,y_test) = datasets.mnist.load_data()
    #数据转换为张量，并归一化
    x_train=2*tf.convert_to_tensor(x_train,dtype=tf.float32)/255 - 1
    x_test=2*tf.convert_to_tensor(x_test,dtype=tf.float32)/255 - 1
    y_train=tf.convert_to_tensor(y_train,dtype=tf.int32)
    y_test=tf.convert_to_tensor(y_test,dtype=tf.int32)

    #y进行one-hot编码
    y_train=tf.one_hot(y_train,depth=10,on_value=None,off_value=None)
    y_test=tf.one_hot(y_test,depth=10,on_value=None,off_value=None)

    #数据重新聚合
    train_dataset =tf.data.Dataset.from_tensor_slices((x_train,y_train))
    test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))

    #数据集切片
    train_dataset=train_dataset.batch(200)
    test_dataset=test_dataset.batch(200)

    #设置学习率
    # optimizer=optimizers.SGD(learning_rate=0.001)

    #设置精度标尺
    # acc_meter=metrics.Accuracy()

    return train_dataset , test_dataset  


#获取单层网路参数
def net(in_ , out_):
    w=tf.Variable(tf.random.truncated_normal([in_ , out_], stddev=0.1))
      
    b=tf.Variable(tf.zeros([out_]))
    return w , b

#获取多层网络参数
def get_netvar(in_1 , in_2 , in_3 , out):
    w1  , b1 = net(in_1 , in_2)
    w2  , b2 = net(in_2 , in_3)
    w3  , b3 = net(in_3 ,  out)
    return w1 , b1  , w2  , b2 ,  w3 , b3 



def train_test (train_dataset ,test_dataset, lr , epoch    ):
    w1 , b1  , w2  , b2 ,  w3 , b3 = get_netvar(784,256,128,10)
    epoch_sumloss = []
    epoch_acc = []
    for inter in range(epoch):
        sum_loss=[]
        sum_acc=[]
        total_correct=0
        #训练，更新网络参数
        for step,(x,y) in enumerate(train_dataset):
            #记录下所有可训练变量的梯度，以便下一步更新
            with tf.GradientTape() as tape:
                x=tf.reshape(x,(-1,28*28))

                h1 =tf.nn.relu( x@w1+b1)
                h2 = tf.nn.relu( h1@w2+b2 )
                out_ = h2@w3+b3
                
                #计算单个方差
                loss=tf.square(out_-y)
                #张量求和，也就是计算所有图片的均方误差和，计算均方差
                loss=tf.reduce_mean(loss)
                #记录loss
                sum_loss.append(loss.numpy())
            #获取每个变量对应的梯度值
            grads=tape.gradient(loss,[w1 , b1 , w2 , b2 , w3 , b3 ])

            #更新网络参数
            w1.assign_sub(lr*grads[0])
            b1.assign_sub(lr*grads[1])
            w2.assign_sub(lr*grads[2])
            b2.assign_sub(lr*grads[3])
            w3.assign_sub(lr*grads[4])
            b3.assign_sub(lr*grads[5])
        epoch_sumloss.append(sum(sum_loss)/ len(sum_loss))
        #测试准确率
        for step,(x,y) in enumerate(test_dataset):
                x=tf.reshape(x,(-1,28*28))
                h1 =tf.nn.relu( x@w1+b1)
                h2 = tf.nn.relu( h1@w2+b2 )
                out_ = h2@w3+b3
                #选取概率最大的类别
                pred=tf.argmax(out_ , axis=1)
                #one-hot编码逆过程
                y=tf.argmax(y,axis=1)
                #比较两者结果是否相等
                correct = tf.equal(pred , y)
                # print(correct)
                total_correct=tf.reduce_sum(tf.cast(correct , dtype=tf.int32)).numpy()
                # print(total_correct , len(correct),total_correct/len(correct))
                sum_acc.append(total_correct/len(correct))
                
        epoch_acc.append(sum(sum_acc)/len(sum_acc))

        print('epoch:' , inter , 'step' , step , 'loss:'  , loss.numpy() , 'acc:' ,sum(sum_acc)/len(sum_acc) )

    return  epoch_sumloss  , epoch_acc

    
def draw(epoch_sumloss , epoch_acc):
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

if __name__ == '__main__' :
    train_dataset , test_dataset = _init()
    epoch_sumloss  , epoch_acc = train_test(train_dataset , test_dataset , 0.01 , 40)
    draw(epoch_sumloss , epoch_acc)
