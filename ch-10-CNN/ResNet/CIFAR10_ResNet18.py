import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , Sequential , datasets , optimizers,losses
import matplotlib.pyplot as plt

#构建Skip Connection 1x1卷积层残差模块类
#创建一个新类，在初始化阶段创建残差块中需要的卷积层，激活函数等
#创建F(x)卷积层
class BasicBlock(layers.Layer):
    #残差模块类
    def __init__(self , filter_num , stride = 1):
        super(BasicBlock , self).__init__()
        #f(x)包含了2个普通卷积层
        #创建卷积层1
        self.conv1 = layers.Conv2D(filter_num , kernel_size=3,strides=stride , padding='SAME')
        #参数标准化层
        self.bn1 = layers.BatchNormalization( )
        #激活函数层
        self.relu = layers.Activation('relu')
        #创建卷积层2
        self.conv2 = layers.Conv2D(filter_num , kernel_size=3 ,strides=1 , padding='SAME')
        self.bn2 = layers.BatchNormalization()
        #identity(x)卷积层，当F(x)的形状与x不相同时，需要进行形状转换
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsample=lambda x: x
    
    def call(self , inputs , training=None):

        out = self.conv1(inputs)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn2(out)
        identity = self.downsample(inputs)

        output = layers.add([out , identity])
        output = tf.nn.relu(output)
        return output



#实现通用ResNet网络模型
class ResNet(keras.Model):
    #新建多个残差模块类
    def build_resblock(self , filter_num , blocks , stride=1):
        #多个残差模块创建容器
        res_blocks = Sequential()
        #第一个BasicBlock的步长可能不为1 ， 实现下采样
        res_blocks.add(BasicBlock(filter_num , stride))

        #其他basicblock步长都为1
        for _ in range(1,blocks):
            res_blocks.add(BasicBlock(filter_num , stride=1))
        return res_blocks
    def __init__(self , layer_dims , num_classes=10):
        super(ResNet , self).__init__()
        #根网络，预处理
        self.stem = Sequential([
            layers.Conv2D(64,kernel_size=3 , strides=1),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=2 , strides=1,padding='same')
        ])
        self.layers1 = self.build_resblock(64 , layer_dims[0])
        self.layers2 = self.build_resblock(128 , layer_dims[1],stride=2)
        self.layers3 = self.build_resblock(256 , layer_dims[2],stride=2)
        self.layers4 = self.build_resblock(512 , layer_dims[3],stride=2)
        #通过pooling层讲高宽降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        #全连接层进行分类
        self.fc = layers.Dense(num_classes)
    def call(self , inputs , training=None):
        x = self.stem(inputs)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

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
    
    #查看标签可知，其为二维数据，将其转换为一维数据
    y_train = tf.squeeze(y_train , axis=1)
    y_test = tf.squeeze(y_test , axis=1)
    # print(y_train)
    #构建训练集对象，随机打乱，预处理，批量化
    train_db = tf.data.Dataset.from_tensor_slices((x_train , y_train))
    train_db=train_db.shuffle(1000).map(preprocess).batch(128)
    #构建测试机对象，预处理，批量化
    test_db = tf.data.Dataset.from_tensor_slices((x_test , y_test))
    test_db=test_db.shuffle(1000).map(preprocess).batch(128)

    return train_db ,test_db

#训练和测试
def train_test(model , train_dataset  ,test_dataset, epochs):
    #交叉熵损失函数类实例，带softmax函数
    CC_loss = losses.CategoricalCrossentropy(from_logits=True)
    #梯度下降优化
    optimizer = optimizers.Adam(learning_rate=0.001)
    #记录每个epoch的损失函数和准确率
    epochs_all_loss = []
    epochs_all_acc = []
    for epoch in range(epochs):
        losses_ = []
        for step , (x , y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # x = tf.expand_dims(x , axis =3)
                out = model(x,training=True)
                y = tf.one_hot(y,depth=10 , on_value=None , off_value=None)
                loss = CC_loss(y , out)
                aver_loss = tf.reduce_mean(loss)
                losses_.append(float(aver_loss))
            grads = tape.gradient(loss , model.trainable_variables)
            optimizer.apply_gradients(zip(grads , model.trainable_variables))
            if step%200==0:
                print("training:epoch",epoch,"step",step , 'loss',float(aver_loss) )
        aver_epoch_loss = tf.reduce_mean(losses_)
        epochs_all_loss.append(aver_epoch_loss)
        
        
        
        correct = 0
        total_samples =0
        for step , (x,y) in enumerate(test_dataset):
            # x = tf.expand_dims(x , axis =3)
            out = model(x,training=False)
            pred = tf.argmax(out , axis=-1)
            y = tf.cast(y , tf.int64)
            correct +=float( tf.reduce_sum( tf.cast( tf.equal(pred , y) , tf.float32 ) ) )
            total_samples += x.shape[0] 
        acc = correct / total_samples
        print('epoch:' , epoch , 'loss:' , float(aver_epoch_loss) , 'acc:' , float(acc),'\n')
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


#resnet18网络模型
def resnet18( ):
    return ResNet([2,2,2,2])
#resnet34网络模型
def resnet34( ):
    return ResNet([3,4,5,3])

#主函数
if __name__ == '__main__' :
    train_dataset , test_dataset = __init()
    resnet_18= resnet18()
    epoch_acc  , epoch_sumloss = train_test(resnet_18,train_dataset ,test_dataset , 10)
    draw(epoch_sumloss , epoch_acc)






