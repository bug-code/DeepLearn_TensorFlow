from tensorflow.keras import layers , Sequential
import tensorflow as tf

'''test 1'''
# network = Sequential([
#     layers.Dense(3,activation=None),
#     layers.ReLU(), 
#     layers.Dense(2,activation=None),
#     layers.ReLU()
# ])

# x = tf.random.normal([4,3])
# out = network(x)
# print(out)

'''
test 2
'''

#layers.Dense(x)表示的是在该网络层中设置x个神经元数量

layers_num = 2
network = Sequential([])
for i in range(layers_num):
    network.add(layers.Dense(5))
    network.add(layers.ReLU())

network.build(input_shape=(4,4))
network.summary()
x = tf.random.normal([4,4])
out = network(x)
print(x)