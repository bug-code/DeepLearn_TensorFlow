import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers , Sequential , metrics

'''
keras提供一些常用的测量工具。
位于keras.metrics模型中专门用于统计训练过程中常用的指标数据

keras测量工具使用方法:
                        1、新建测量器
                        2、写入数据
                        3、读取统计数据
                        4、清零测量器
'''

'''
以下注释部分即为常规测量器的实现，将该代码插入在需要的代码中
'''
# #新建平均测量器，测量loss数据
# loss_meter= metrics.Mean()
# #在每个训练step后写入数据
# loss_meter.update_state(float(loss))
# if !(step % 100):
#     #每若干个step后输出统计数据（可有可无）
#     print(step , 'loss:' ,loss_meter.result())
#     #测量器清零
#     loss_meter.reset_states()

#如果使用的是准确率测量器
#则在写入数据时，需要将预测值和实际值一同写入测量其中
#acc_meter.update_state(y,pred)

