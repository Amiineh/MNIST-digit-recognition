import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("mnist", one_hot=True)
batch_size = 100
learning_rate = 0.1
w_stddev = 0.2
hidden_layer_size = 10
num_iteration = 10000
early_stopping_threshold = 500

def ValidationAccuracy(sess, validation_data, x, y):
    ''' This function returns accuracy on validation set for early stopping '''
    correct_results = tf.equal(tf.argmax(mnist.validation.labels, 1), tf.argmax(validation_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_results, tf.float32))
    return sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

with tf.name_scope("input"):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="x")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="labels")

with tf.name_scope("first_layer"):
    W1 = tf.Variable(tf.random_normal(shape=(784, hidden_layer_size), stddev=w_stddev, mean=0), name="W1")
    b1 = tf.Variable(tf.zeros(hidden_layer_size), name="b1")
    h = tf.sigmoid(tf.matmul(x, W1)+b1, name="h")
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("biases", b1)
    tf.summary.histogram("activations", h)

with tf.name_scope("second_layer"):
    W2 = tf.Variable(tf.random_normal(shape=(hidden_layer_size, 10), stddev=w_stddev, mean=0), name="W2")
    b2 = tf.Variable(tf.zeros(10), name="b")
    output = tf.add(tf.matmul(h, W2),b2, name="output")
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("biases", b2)
    tf.summary.histogram("activations", output)

with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

with tf.name_scope("train"):
    optimiser = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimiser.minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_results = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_results, tf.float32))

merge = tf.summary.merge_all()

# saver = tf.train.Saver()
# saver.restore(sess=sess, save_path='./save')

def train(sess, writer):
    ''' Train the network for 'num_iteration' times'''
    global x, y, W1, b1, W2, b2
    for i in range(num_iteration):
        image, label = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: image, y: label})
        # if i%100==0:
            # saver.save(sess=sess, save_path='./save')
        loss, b=sess.run((cross_entropy, merge), feed_dict={x:image, y:label})
        smry = tf.Summary(value=[tf.Summary.Value(tag="loss_learning_rate", simple_value=loss)])
        writer.add_summary(smry, i)

    print "accuracy on training data: ", sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
    print "accuracy on test data: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

def Early_stopping(sess, writer):
    ''' In this function we start training the network,
        and if the validation accuracy hasn't changed in the past 'early_stopping_threshold' iterations,
        we have reached an acceptable accuracy and stop training. '''
    global x, y, W1, b1, W2, b2
    validation_acc_max, W1_max, b1_max, W2_max, b2_max, epoch_max = 0, W1, b1, W2, b2, 0
    for i in range(num_iteration):
        image, label = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: image, y: label})
        if i%50==0:
            loss, b=sess.run((cross_entropy, merge), feed_dict={x:image, y:label})
            smry = tf.Summary(value=[tf.Summary.Value(tag="loss_early_stopping", simple_value=loss)])
            writer.add_summary(smry, i)

        if i%10 == 0:
            new_validation = ValidationAccuracy(sess, output, x, y)
            if new_validation > validation_acc_max:
                validation_acc_max , W1_max , b1_max, W2_max, b2_max, epoch_max = new_validation, W1, b1, W2, b2, i
        if i > 500 + epoch_max:
            print "Early stopping..."
            W1, b1, W2, b2 = W1_max, b1_max, W2_max, b2_max,
            break

    print "Final accuracy on test data: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

def CV_5fold(sess):
    ''' In this function, we use 5-Fold Cross Validation and average the results of each fold to avoid over-fitting. '''
    global x, y, W1, b1, W2, b2
    n = len(mnist.train.images)
    step = n // 5
    batch_size = 100
    acc_valid_sum, acc_test_sum = 0, 0
    for j in range(0, n, step):
        sess.run(tf.global_variables_initializer())
        x_valid, y_valid = mnist.train.images[j:j+step], mnist.train.labels[j: j+step]
        x_train, y_train = (np.concatenate((mnist.train.images[0:j], mnist.train.images[j+step:])), np.concatenate((mnist.train.labels[0:j], mnist.train.labels[j+step:])))

        for _ in range(5):
            for k in range(0, len(x_train), batch_size):
                sess.run(train_op, feed_dict={x: x_train[k:k+batch_size], y: y_train[k:k+batch_size]})

        acc_valid_sum += sess.run(accuracy, feed_dict={x: x_valid , y: y_valid})
        acc_test_sum += sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

    acc_test_sum /= 5
    acc_valid_sum /= 5

    print "accuracy on test data: ", acc_test_sum, "\naccuracy on validation data: ", acc_valid_sum


if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    writer = tf.summary.FileWriter('./log/', sess.graph)

    train(sess, writer)

    sess.run(tf.global_variables_initializer())
    Early_stopping(sess, writer)

    sess.run(tf.global_variables_initializer())
    CV_5fold(sess)

