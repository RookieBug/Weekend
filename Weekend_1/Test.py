#coding:utf-8

#测试在泰坦尼克号上的生存概率

#输入性别、年龄、客舱等级、兄弟姐妹和配偶所在数量、父母子女所在数量、船票价格

#最终输出结果是：死亡概率，生存概率

import tensorflow as tf
import pandas as pd

def add_layer(layer_name,in_data,in_size,out_size,activation_funcation=None):
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'Weights')
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name = 'biases')
        Wx_plus_b = tf.matmul(in_data,Weights) + biases
    
    if( activation_funcation is None ):
        out_data = Wx_plus_b
    else:
        out_data = activation_funcation(Wx_plus_b)
    
    return out_data

# data = pd.read_csv('Titanic/test.csv')
# data = data.fillna(0)
# data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
# dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()

x = [[1,20,1,1,2,30]]

#print(dataset_X)
X = tf.placeholder(tf.float32, [None,6])
layer_1 = add_layer('layer_1',X, 6, 3, activation_funcation = None)
y_pred = add_layer('layer_2',layer_1,3,2,activation_funcation = tf.nn.softmax)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'my_net_0.833333/model.ckpt')
    result = sess.run(y_pred, feed_dict={X:x})
    print(result)