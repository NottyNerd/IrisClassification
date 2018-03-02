import pandas as pd
import numpy as np
import matplotlib.pyplot as mpt
import tensorflow as tf



#load data
columns =['sepal length','sepal width','petal length','petal width']
output_columns = ['Iris-setosa','Iris-versicolor','Iris-virginica']
dataframe = pd.read_csv('data/Iris_Dataset.csv')
#print(dataframe)

#introduce labels

#predict next value in continuous set

#prepare to tensorflow

#tensors are generic versions of vectors and matrices

inputX = dataframe.loc[:,columns].astype(float).as_matrix()

dataframe.Class = pd.Categorical(dataframe.Class)
dataframe['cls'] = dataframe.Class.cat.codes
inputY =dataframe.loc[:,['cls']].as_matrix()
print(inputX)
print(inputY)

#write hyperparameters like learning rate(how fast do we reach convergence)

learning_rate = 0.000001
training_epochs = 2000
display_stacks = 50
n_samples = inputX.size

#Create computation graph/NN
#placeholders are gateways for data into CG
x = tf.placeholder(tf.float32,[None,4])

#weight
w = tf.Variable(tf.zeros([4,4]))
print(w)
#bias
b = tf.Variable(tf.zeros([4]))
print(b)
#multiply weights by input to govern how data flows in computation graph
y_values = tf.add(tf.matmul(x,w), b)

#apply softmax to normalize values by converting to a probability
y = tf.nn.softmax(y_values)
#feed in a matrix of labels
y_ =tf.placeholder(tf.float32, [None, 1])

#train

cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY})

    if (i) % display_stacks == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print("training cost " + str((i)) + "cost: "+ "{:.9f}".format(cc) )
print("finished" )
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print("training cost= "+ str(training_cost) + " W= "+ str(sess.run(w))+" b= "+ str(sess.run(b)) )