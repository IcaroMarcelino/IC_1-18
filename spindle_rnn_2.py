from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import csv
import numpy as np
import random
import time


def initial_population(excerpt):
    csvfile = open('DREAMS_Sleep_Spindles/ex' + str(excerpt) + '.csv','r')
    data = csv.reader(csvfile)

    train_input = []
    train_output = []

    for row in data:
        train_input.append([float(x) for x in row[1:9]])
        train_output.append([int(row[10]), not(int(row[10]))])


    csvfile.close()

    test_input = train_input[-int(0.3*len(train_input)):]
    test_output = train_output[-int(0.3*len(train_output)):]

    train_input = train_input[:-int(0.3*len(train_input))]
    train_output = train_output[:-int(0.3*len(train_output))]

    return train_input, train_output, test_input, test_output

def split_data_set(percent_test, train_input, train_output):
    test_in = train_input[-int(percent_test*len(train_input)):]
    test_out = train_output[-int(percent_test*len(train_output)):]

    train_in = train_input[:-int(percent_test*len(train_input))]
    train_out = train_output[:-int(percent_test*len(train_output))]

    return train_in, train_out, test_in, test_out

def shuffle_data_set(input_data, labels):
    temp = list(zip(input_data, labels))
    random.shuffle(temp)
    input_data, labels = zip(*temp)

    return input_data, labels

def transformed_data_set(excerpt, window_size, shuffle):
    csvfile = open('DREAMS_Sleep_Spindles/ex' + str(excerpt) + '.csv','r')
    dt = csv.reader(csvfile)

    data = []
    for row in dt:
        data.append(row)

    csvfile.close()

    train_input = []
    train_output = []

    n_batches = int(len(data)/window_size)

    for i in range(n_batches):

        lo = i*(window_size) 
        up = (i+1)*window_size

        temp = data[lo:up]

        train_input.append([])
        train_output.append([0, 1])

        for row in temp:
            train_input[i].append([float(x) for x in row[1:10]])

            if int(row[10]) == 1:
                train_output[i] = [1, 0]

    if shuffle:
        train_input, train_output = shuffle_data_set(train_input, train_output)
           
    return train_input, train_output

def save_scores(best_scores, losses, times, path, filename, nexec):
    output = open(path + filename + "_" + str(nexec) + ".csv", 'w')
    
    wr = csv.writer(output)
    for i in list(range(len(best_scores))):
        wr.writerow([i, best_scores[i], losses[i], times[i]])

    output.close()
    return

def RNN(x, weights, biases, timesteps, num_hidden):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def execute(nexec, learning_rate, training_steps, batch_size, display_step, filename):
    tf.reset_default_graph()

    timesteps = 5 # timesteps

    train_in = []
    train_out = []

    for i in range(1,9):
        temp_in, temp_out = transformed_data_set(i, timesteps, 0)
        train_in  += temp_in
        train_out += temp_out

    train_in, train_out = shuffle_data_set(train_in, train_out)
    train_in, train_out, test_in, test_out = split_data_set(.3, train_in, train_out)

    # Network Parameters
    num_input = 9
    num_hidden = 1024 # hidden layer num of features
    num_classes = 2
     
    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }


    logits = RNN(X, weights, biases, timesteps, num_hidden)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        scores = []
        times = []
        losses = []
        i = 0
        for step in range(1, int(training_steps)+1):
            if (i+1)*batch_size > len(train_in):
                i = 0

            batch_x = train_in[(i*batch_size):((i+1)*batch_size)]
            batch_y = train_out[(i*batch_size):((i+1)*batch_size)]

            # Run optimization op (backprop)
            init = time.time()
            
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})

                scores.append(acc)
                losses.append(loss)
                #print("Step " + str(step) + ", Minibatch Loss= " + \
                #      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                #      "{:.3f}".format(acc))
            end = time.time()

            i += 1
            times.append(end-init)

        save_scores(scores, losses, times, "Scores/SCR_", filename, nexec)

        print("Optimization Finished!")

        # test_data = np.array(test_in).reshape((batch_size, timesteps, num_input))
        # test_label = test_out
        print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: test_in, Y: test_out}))

# Training Parameters
learning_rate = 1e-5
training_steps = 100000
batch_size = 100
display_step = 1
filename = "RNN_LR5_H1024"

for i in range(0,50):
	execute(nexec = i, 
    	    learning_rate = learning_rate, 
        	training_steps = training_steps, 
        	batch_size = batch_size, 
        	display_step = display_step, 
        	filename = filename)

