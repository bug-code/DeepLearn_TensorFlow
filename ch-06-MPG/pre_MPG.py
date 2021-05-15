import tensorflow as tf
from tensorflow import losses
import tensorflow.keras as keras
from tensorflow.keras import layers ,optimizers,datasets
import pandas as pd
import matplotlib.pyplot as plt

def data_init():
    #在线下载汽车效能数据集
    dataset_path = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    #利用pandas读取数据集，字段  效能、气缸数、排量、马力、重量、加速度、型号年份、产地
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
    #将该数据集组合成CSV数据集，从dataset_path中下载，属性名为list中的名字，属性中缺少值的用？表示，使用\t对齐，以空格分隔，跳过开头初始空格
    raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values="?",comment='\t',sep=" ",skipinitialspace=True)

    dataset = raw_dataset.copy()

    # print(dataset.head())

    '''清除处理数据集中的空字段数据项
    '''
    #统计空白数据项
    dataset.isna().sum()
    #删除空白数据项
    dataset=dataset.dropna()
    #再次统计空白数据项
    dataset.isna().sum()

    '''Origin字段为类别类型数据，将其转换为类似于one-hot编码
    从数据集中单独取出origin属性数据
    在分离Origin后数据集新添加三个属性：USA 、 Europe、Japan。使用来源于该产地则该属性项下赋值为1，否则赋值为0
    '''

    Origin = dataset.pop('Origin')

    #为处理后的数据集分别添加三个属性项
    dataset['USA']=(Origin==1)*1.0
    dataset['Europe']=(Origin==2)*1.0
    dataset['Japan']=(Origin==3)*1.0

    #查看数据集之间的两两关系，绘制图像
    def draw_relation(data):
        data_mat = data.transpose().values
        x_lables = ['Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year']
        plt.figure(figsize=(16,10))
        for i in range(6):
            i +=1
            plt.subplot(2,3,i)
            plt.ylabel('MPG')
            plt.xlabel(x_lables[i-1])
            plt.scatter(data_mat[i],data_mat[0],c='red',marker='*')
        plt.show()
    draw_relation(dataset)


    '''切分数据集'''
    #随机抽取原始数据集中的80%作为训练集
    train_dataset = dataset.sample(frac=0.8 , random_state=0)
    #原始数据集去除训练集得到测试集
    test_dataset = dataset.drop(train_dataset.index)

    #分离出训练集和测试集需要进行预测的标签
    train_labels  =  train_dataset.pop('MPG')
    test_labels   = test_dataset.pop('MPG')

    '''
    数据标准化
    '''
    #获得数据集中各字段均值和标准差等属性
    train_stats = train_dataset.describe().transpose()
    test_stats = test_dataset.describe().transpose()
    #归一化数据集
    #减去每个字段的均值，并除以标准差
    train_dataset=(train_dataset-train_stats['mean'])/train_stats['std']
    test_dataset=(test_dataset-test_stats['mean'])/test_stats['std']
    # print(train_dataset,'\n',train_labels , '\n',test_dataset,'\n',test_labels)

    #构建dataset对象
    train_db=tf.data.Dataset.from_tensor_slices((train_dataset.values , train_labels.values))
    test_db=tf.data.Dataset.from_tensor_slices((test_dataset.values , test_labels.values))

    #随机打散，批量化
    train_db=train_db.shuffle(100).batch(32)
    test_db=test_db.shuffle(100).batch(32)
    return train_db ,  test_db

#创建网络
class net(keras.Model):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=layers.Dense(64,activation='relu')
        self.fc2=layers.Dense(64,activation='relu')
        self.fc3=layers.Dense(1)
    def call(self,in_,training=None,mask=None):
        x = self.fc1(in_)
        x = self.fc2(x)
        x = self.fc3(x)

        return x



#训练测试
train_db ,  test_db = data_init()
model=net()
model.build(input_shape=(4,9))
# print(model.summary)
optimizers=tf.keras.optimizers.RMSprop(0.001)
train_epoch_mse=[]
test_epoch_mse=[]
for epoch in range(200):
    train_mse=[]
    test_mse=[]
    for step , (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out=model(x)
            loss=tf.reduce_mean(losses.mae(y,out))
            # loss=tf.reduce_mean(losses.MSE(y,out))
            train_mse.append(loss)
        if step%10==0:
            print("epoch:",epoch,"step:",step,"loss:",float(loss))
        grads=tape.gradient(loss,model.trainable_variables)
        optimizers.apply_gradients(zip(grads,model.trainable_variables))

    train_epoch_mse.append(sum(train_mse)/len(train_mse))
    for step , (x,y) in enumerate(test_db):
        with tf.GradientTape() as tape:
            out=model(x)
            loss=tf.reduce_mean(losses.mae(y,out))
            # loss=tf.reduce_mean(losses.MSE(y,out))
            test_mse.append(loss)
       
    test_epoch_mse.append(sum(test_mse)/len(test_mse))

plt.xlabel('epoch')
plt.ylabel('MAE')
plt.plot(train_epoch_mse,color='blue',label='train')
plt.plot(test_epoch_mse,color='red',label='test')
plt.legend()
plt.show()


