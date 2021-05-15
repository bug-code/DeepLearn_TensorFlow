import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers , Sequential , losses , metrics , optimizers , datasets


tf.random.set_seed(2345)
#自定义数据预处理函数
def preprocess(x , y):
    #标准化到[0-1]
    #x = tf.cast(x , dtype=tf.float32)/255.
    #标准化到[-1,1]之间
    x = 2 * tf.cast(x , tf.float32)/255. - 1
    #打平
    y = tf.cast(y , tf.int32)
    return x , y

#数据初始化
def __init( ):
    """
    加载数据集
    """
    (x_train , y_train) , (x_test , y_test) = datasets.cifar10.load_data()
    # print(x_train.shape)
    #查看标签可知，其为二维数据，将其转换为一维数据
    y_train = tf.squeeze(y_train , axis=1)
    y_test = tf.squeeze(y_test , axis=1)
    #构建训练集对象，随机打乱，预处理，批量化
    train_db = tf.data.Dataset.from_tensor_slices((x_train , y_train))
    train_db=train_db.shuffle(1000).map(preprocess).batch(128)
    #构建测试机对象，预处理，批量化
    test_db = tf.data.Dataset.from_tensor_slices((x_test , y_test))
    test_db=test_db.shuffle(1000).map(preprocess).batch(128)

    return train_db ,test_db

#训练和测试
def train_test(first_net , second_net , train_dataset  , test_dataset, epochs):
    #交叉熵损失函数类实例，带softmax函数
    CC_loss = losses.CategoricalCrossentropy(from_logits=True)
    #Adam优化
    optimizer = optimizers.Adam(learning_rate=1e-4)

    #列表合并，合并两个自网络的参数
    variables = first_net.trainable_variables + second_net.trainable_variables


    #记录每个epoch的损失函数和准确率
    epochs_all_loss = []
    epochs_all_acc = []
    for epoch in range(epochs):
        losses_ = []
        for step , (x , y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                out1 = first_net(x)
                out1 = tf.squeeze(out1 , axis=1)
                out1 = tf.squeeze(out1 , axis=1)
                # print(out1[0],'\n')
                out = second_net(out1)
                #只需在训练集中进行one—hot编码
                y  = tf.one_hot(y , depth=10,on_value=None , off_value=None)
                loss = CC_loss(y , out)
                # print(loss)
                aver_loss = tf.reduce_mean(loss)
                losses_.append(float(aver_loss))
                
            grads = tape.gradient(loss , variables)
            optimizer.apply_gradients(zip(grads , variables))
            if step%200==0:
                print("training:epoch",epoch,"step",step , 'loss',float(aver_loss),'\n' )
        aver_epoch_loss = tf.reduce_mean(losses_)
        epochs_all_loss.append(aver_epoch_loss)
        
        
        
        correct = 0
        total_samples =0
        for step , (x,y) in enumerate(test_dataset):
            # print(x)
            # x = tf.expand_dims(x , axis =3)
            out1_test = first_net(x)
            out= second_net(out1_test)
            pred = tf.argmax(out , axis=-1)
            # tf.argmax(y,axis=1)
            y = tf.cast(y , tf.int64)
            correct +=float( tf.reduce_sum( tf.cast( tf.equal(pred , y) , tf.float32 ) ) )
            total_samples += x.shape[0] 
            # print("testing:epoch",epoch,"step",step,'\n')

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

def create_model( ):
    '''
    创建改进的VGG-13网络模型
    '''

    #创建卷积网络网络
    conv_layers = [
        #创建包含多网络层的列表

        #conv-conv-pooling 单元1
        #64个3x3卷积核，输入输出同大小
        layers.Conv2D(64,kernel_size=3 , padding='same' , activation=tf.nn.relu) ,
        layers.Conv2D(64,kernel_size=3 , padding='same' , activation=tf.nn.relu) ,
        #池化层高宽减半
        layers.MaxPooling2D(pool_size=2 , strides=2 , padding='same') , 
        
        #conv-conv-pooling 单元2
        #128个3x3卷积核，输入输出同大小
        layers.Conv2D(128,kernel_size=3 , padding='same' , activation=tf.nn.relu) ,
        layers.Conv2D(128,kernel_size=3 , padding='same' , activation=tf.nn.relu) , 
        #池化层高宽减半
        layers.MaxPooling2D(pool_size=2 , strides=2 , padding='same') , 
        
        #conv-conv-pooling 单元3
        #256个3x3卷积核，输入输出同大小
        layers.Conv2D(256 , kernel_size=3 , padding='same' , activation=tf.nn.relu) , 
        layers.Conv2D(256 , kernel_size=3 , padding='same' , activation=tf.nn.relu) , 
        #池化层高宽减半
        layers.MaxPooling2D(pool_size=2 , strides=2 , padding='same') , 
        
        #conv-conv-pooling 单元4
        #64个3x3卷积核，输入输出同大小
        layers.Conv2D(512 , kernel_size=3 , padding='same' , activation=tf.nn.relu) ,
        layers.Conv2D(512 , kernel_size=3 , padding='same' , activation=tf.nn.relu) ,
        #池化层高宽减半  
        layers.MaxPooling2D(pool_size=2 , strides=2 , padding='same') ,

        #conv-conv-pooling 单元5
        #64个3x3卷积核，输入输出同大小
        layers.Conv2D(512 , kernel_size=3 , padding='same' , activation=tf.nn.relu) ,
        layers.Conv2D(512 , kernel_size=3 , padding='same' , activation=tf.nn.relu) ,
        #池化层高宽减半
        layers.MaxPooling2D(pool_size=2 , strides=2 , padding='same') ,
    ]

    #创建全连接神经网络
    fc_net = Sequential([
        layers.Dense(256 , activation=tf.nn.relu) , 
        layers.Dense(128 , activation=tf.nn.relu) , 
        layers.Dense(10 , activation=None) , 
    ])

    #利用层列表创建网络容器
    conv_net = Sequential(conv_layers)

    #buid 2个自网络
    conv_net.build(input_shape=[None,32,32,3])
    fc_net.build(input_shape=[None , 512])

    # conv_net.summary()
    # fc_net.summary()
    return conv_net , fc_net


#主函数
if __name__ == '__main__' :
    train_dataset , test_dataset = __init()
    conv_net , fc_net = create_model()
    epoch_acc  , epoch_sumloss = train_test(conv_net , fc_net,train_dataset ,test_dataset , 10)
    draw(epoch_sumloss , epoch_acc)