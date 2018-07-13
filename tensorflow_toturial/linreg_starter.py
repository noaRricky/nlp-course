""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

# import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow_toturial import utils

DATA_FILE = 'birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create data set and iterator
# Remember both X and Y are scalars with type float
data_set = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))
iterator = data_set.make_initializable_iterator()
X, Y = iterator.get_next()

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
w = tf.get_variable(name='weight', shape=(
    1,), initializer=tf.zeros_initializer())
b = tf.get_variable(name='bias', shape=(
    1,), initializer=tf.zeros_initializer())

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = w * X + b

# Step 5: use the square error as the loss function
loss = tf.square(Y_predicted - Y, name='loss')

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.001).minimize(loss)

start = time.time()

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    # Create a filewriter to write the model's graph to TensorBoard
    writer = tf.summary.FileWriter('./graph/linear_dataset', sess.graph)

    # Step 8: train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            # Execute train_op and get the value of loss.
            # Don't forget to feed in data for placeholders
            _, cur_loss = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += cur_loss
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])

print('Took: %f seconds' % (time.time() - start))

# plot the results
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r',
         label='Predicted data with squared error')
# plt.plot(data[:,0], data[:,0] * (-5.883589) + 85.124306, 'g', label='Predicted data with Huber loss')
plt.legend()
plt.show()
