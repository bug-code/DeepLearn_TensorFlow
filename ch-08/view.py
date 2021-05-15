import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers , Sequential  , losses , metrics, models

'''
在网络训练过程中，通过web端远程监控网络的训练进度，可视化网络的训练结果。
可视化工具:需安装tensorboard
          需要模型代码和浏览器相互配合

visdom工具能够提供更加丰富的可视化方式，同时实时性更高。
能够直接支持pytorch张量类型数据，但是该工具不支持tensorflow张量，
需要转换为numpy数组
'''

'''
模型端：
        1、创建写入监控数据的summary类 ，在需要的时候写入监控数据
            使用tf.summary.create_file_writer创建监控对象类实例，
            并指定监控数据的写入目录。
            每类数据通过字符串名区分，同类数据写入同名数据库
'''
#创建监控类，监控数据写入log_dir目录
summary_writer = tf.summary.create_file_writer(log_dir)
#设置写入环境
with summary_writer.as_default():
    #当时间戳step上的数据有loss,写入到名为train-loss数据库中
    tf.summary.scalar('train-loss' , float(loss) , step=step)
    #当时间戳step上的数据有loss,写入到名为test-acc数据库中
    tf.summary.scalar('test-acc' , float(acc),step=step)
    #对于图片类型数据，通过tf.summary.image()函数写入监控数据 ，
    #  max-output表示一次最多能展示9张图片
    tf.summary.image("val-onebyone-images" , val_images , 
    max_outputs=9 , step=step)
    #可视化真实标签的直方分布图
    tf.summary.histogram('y-hist' , y , step=step)
    #查看文本信息
    tf.summary.text('loss-text' , str(float(loss))


'''
浏览器端：
        tensorboard自持通过tf.summary.histogram查看张量数据直方分布图
        通过tf.summary.text打印文本信息等功能
        1、打开web后端:
                    在cmd终端运行tensorboard --logdir path 
                    指定web后端监控的文件目录path
        2、打开浏览器,网址http://localhost:6006 即可监控网络训练进度
'''

