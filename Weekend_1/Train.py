#coding:utf-8

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

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
#读取数据
data = pd.read_csv('Titanic/train.csv')
#将空白全部填充为0
data = data.fillna(0)
#性别，男性为1，女性为0
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
#添加死亡表格（添加后，死亡在前，存活在后）
data['Deceased'] = data['Survived'].apply(lambda s: 1 - s)
#训练集
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()
dataset_Y = data[['Deceased', 'Survived']].as_matrix()
# print(dataset_Y)
# 分割数据集，将数据集分开，可以防止过拟合。
X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y,
                                                  test_size=0.1,
                                                  random_state=42)
#占位符
with tf.name_scope('Data'):
    X = tf.placeholder(tf.float32, [None,6],name='input')
y = tf.placeholder(tf.float32, [None,2])
#print(dataset_X[1])
layer_1 = add_layer('layer_1',X, 6, 3, activation_funcation = None)
y_pred = add_layer('layer_2',layer_1,3,2,activation_funcation = tf.nn.softmax)

cost = tf.reduce_mean(tf.reduce_sum(-y * tf.log(y_pred + 1e-10)),name='cost')
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
 
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
with tf.Session() as sess:
    tf.global_variables_initializer().run()
     
        # training loop
    for epoch in range(1000):
        total_loss = 0.
        for i in range(len(X_train)):
            # prepare feed data and run
            feed_dict = {X: [X_train[i]], y: [y_train[i]]}
            _, loss = sess.run([train_step, cost], feed_dict=feed_dict)
            total_loss += loss
        # display loss per epoch
        if(epoch % 100 == 0):
            print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))

    # Accuracy calculated by TensorFlow
    accuracy = sess.run(acc_op, feed_dict={X: X_val, y: y_val})
    print("Accuracy on validation set: %.9f" % accuracy)
    saver = tf.train.Saver()
    path = 'my_net_' + str(accuracy) + '/model.ckpt'
    saver.save(sess,path)

