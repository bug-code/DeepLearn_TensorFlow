import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers , Sequential , optimizers ,losses , datasets




'''
高级接口使用流程：
                1、模型装配
                    compile函数:指定优化器对象、损失函数类型、评价指标
                2、模型训练
                    fit函数：送如待训练的数据集和验证用的数据集
                    返回训练过程总数据记录：history（包含训练过程中loss,测量指标等）
                3、模型测试
                    Model.predict(x)方法完成模型预测
                    如果只是简单测试模型性能，model.evaluate(db)即可
'''
def _init( ):
    #load数据集
    (x_train,y_train) , (x_test,y_test) = datasets.mnist.load_data()
    #数据转换为张量，并归一化
    x_train=2*tf.convert_to_tensor(x_train,dtype=tf.float32)/255 - 1
    x_train = tf.reshape(x_train , (-1,28*28))
    
    x_test=2*tf.convert_to_tensor(x_test,dtype=tf.float32)/255 - 1
    #需要将原始数据展平
    x_test = tf.reshape(x_test , (-1,28*28))

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

    return train_dataset , test_dataset  


#模型装配
net = Sequential([
    layers.Dense(256 ,activation='relu'),
    layers.Dense(128 ,activation='relu'),
    layers.Dense(64 ,activation='relu'),
    layers.Dense(32 ,activation='relu'),
    layers.Dense(10)   
])
net.build(input_shape=(None,28*28) )
# net.summary()
net.compile(optimizer=optimizers.Adam(learning_rate=0.01) , loss=losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
train_db , test_db  = _init()
#模型训练
his = net.fit(train_db , epochs=20 , validation_data=test_db , validation_freq=20)



#模型测试
x , y =next(iter(test_db))
pre = net.predict(x)
#概率相加为1
pre = layers.Softmax(axis=-1)(pre)
#选取概率最大的索引位置
pre = tf.argmax(pre , axis = -1)
print(pre)
##测试模型性能
eval = net.evaluate(test_db)
print(eval)

'''

#模型保存方式1：张量方式
net.save_weights('weights.cktp')
print('-------------------------save weights.--------------------------')
del net
#复原神经网络模型
net = Sequential([
    layers.Dense(256 ,activation='relu'),
    layers.Dense(128 ,activation='relu'),
    layers.Dense(64 ,activation='relu'),
    layers.Dense(32 ,activation='relu'),
    layers.Dense(10)   
])
net.compile(optimizer=optimizers.Adam(learning_rate=0.01) , loss=losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
#读取模型参数到当前网络
net.load_weights('weights.ckpt')
print('------------------------------loaded weights-------------------------')

'''


'''
#模型保存方式2：网络方式
Model.save(doc)函数将模型结构和模型参数保存到文件上,
keras.models.load_model(doc)即可复原网络结构和网络参数

#保存文件
net.save('model.h5')
print('----------------------------------saved total model------------------')
del net
#复原文件
net = keras.models.load_model('model.h5')
'''



'''
#模型保存方式4：SaveModel方式
SaveModel方式具有平台无关性，支持移动端和网页端等
tf.saved_model.save(network , 'path')将模型以SaveModel方式保存在path目录下
tf.saved_model.load函数恢复模型对象
#保存模型
tf.saved_model.save(net , 'model')
print('-----------------------saving model--------------------------')
del net

#恢复模型
print('--------------------loading model------------------------')
net = tf.saved_model.load('model')
'''

