import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , Sequential , datasets , optimizers,losses
import matplotlib.pyplot as plt
'''
改进LeNet-5神经网络，
1、将输入由32x32([b,32,32,6])改为28x28([b,28,28,6])
2、卷积层[b,28,28,6]改为[6,3,3,6] 即：6个3x3大小的卷积核层
3、下采样层[b,14,14,6]改为最大池化层(高宽各减半的池化层) 
4、卷积层[b , 16,16,6]
5、下采样层改为最大池化层 （高宽各减半的池化层）
6、全连接网络层（120个神经元）
7、全连接网络层（84个神经元）
8、高斯连接层改为全连接层（输出）
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
    # y_test=tf.one_hot(y_test,depth=10,on_value=None,off_value=None)

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



#训练和测试
def train_test(model , train_dataset  ,test_dataset, epochs):
    #交叉熵损失函数类实例，带softmax函数
    CC_loss = losses.CategoricalCrossentropy(from_logits=True)
    #梯度下降优化
    optimizer = optimizers.SGD(learning_rate=0.001)
    #记录每个epoch的损失函数和准确率
    epochs_all_loss = []
    epochs_all_acc = []
    for epoch in range(epochs):
        losses_ = []
        for step , (x , y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x = tf.expand_dims(x , axis =3)
                out = model(x)
                loss = CC_loss(y , out)
                # print(loss)
                aver_loss = tf.reduce_mean(loss)
                losses_.append(float(aver_loss))
            grads = tape.gradient(loss , model.trainable_variables)
            optimizer.apply_gradients(zip(grads , model.trainable_variables))
        aver_epoch_loss = tf.reduce_mean(losses_)
        epochs_all_loss.append(aver_epoch_loss)
        
        
        
        correct = 0
        total_samples =0
        for step , (x,y) in enumerate(test_dataset):
            # print(x)
            x = tf.expand_dims(x , axis =3)
            out = model(x)
            pred = tf.argmax(out , axis=-1)
            # tf.argmax(y,axis=1)
            y = tf.cast(y , tf.int64)
            correct +=float( tf.reduce_sum( tf.cast( tf.equal(pred , y) , tf.float32 ) ) )
            total_samples += x.shape[0] 
        acc = correct / total_samples
        # print('acc' , float(acc))
        print('epoch:' , epoch , 'loss:' , float(aver_epoch_loss) , 'acc:' , float(acc))
        epochs_all_acc.append(acc)

    return epochs_all_acc , epochs_all_loss

        

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


#创建LeNet-5神经网络模型
net = Sequential([
    #卷积层[6,3,3,6]
    layers.Conv2D(6 , kernel_size=3 , strides=1),
    #池化层,高宽减半
    layers.MaxPooling2D(pool_size=2 , strides=2),
    #激活层
    layers.ReLU(),
    #卷积层[16,3,36]
    layers.Conv2D(16 , kernel_size=3 , strides=1),
    layers.MaxPooling2D(pool_size=2 , strides=2),
    layers.ReLU(),
    #展平层，方便全连接层处理
    layers.Flatten(),

    #设置全连接层
    layers.Dense(120,activation='relu'),
    layers.Dense(84,activation='relu'),
    layers.Dense(10),
])

net.build(input_shape=(4,28,28,1))
# net.summary()

if __name__ == '__main__' :
    train_dataset , test_dataset = _init()
    epoch_acc  , epoch_sumloss = train_test(net,train_dataset ,test_dataset , 100)
    draw(epoch_sumloss , epoch_acc)