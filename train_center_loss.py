from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os


TRAIN_SIZE = 50000
TEST_SIZE = 10000
INPUT_SIZE = 28
INPUT_CHANNEL = 1
NUM_CLASSES = 10

NUM_EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 1e-3

ALPHA = 0.5
LAMBDA = 1

DATASET_DIR = './MNIST'
IMAGES_DIR = './images/center_loss'

slim = tf.contrib.slim


def inference(inputs):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            net = slim.conv2d(inputs, num_outputs=32, scope='conv1_1')
            net = slim.conv2d(net, num_outputs=32, scope='conv1_2')
            net = slim.max_pool2d(net, scope='pool1')
            net = slim.conv2d(net, num_outputs=64, scope='conv2_1')
            net = slim.conv2d(net, num_outputs=64, scope='conv2_2')
            net = slim.max_pool2d(net, scope='pool2')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, num_outputs=512, scope='fc1')
            features = slim.fully_connected(net, num_outputs=128, scope='fc2')
            logits = slim.fully_connected(features, num_outputs=NUM_CLASSES, activation_fn=None, scope='fc3')
        return logits, features


def get_center_loss(features, labels):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [NUM_CLASSES, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)

    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss = tf.nn.l2_loss(features - centers_batch)

    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = ALPHA * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op


def display_fig(features, colors, epoch):
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    # reducing feature to 3 dimension
    pca = PCA(n_components=3)
    features = pca.fit_transform(features)

    # color
    colors = colors.tolist()
    color_list = ['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'black', 'gray']
    for i in range(len(colors)):
        colors[i] = color_list[colors[i]]

    # visualizing result
    fig = plt.figure()
    axe = Axes3D(fig)
    axe.scatter(features[:, 0], features[:, 1], features[:, 2], c=colors, marker='.')
    axe.set_xlim(-0.6, 0.6)
    axe.set_ylim(-0.6, 0.6)
    axe.set_zlim(-0.6, 0.6)

    fig.savefig(os.path.join(IMAGES_DIR, '%d.png' % epoch))
    plt.close(fig)


def train():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], name='inputs')
    labels = tf.placeholder(dtype=tf.int64, shape=[None, ], name='labels')
    logits, features = inference(inputs)

    # defining acc
    predictions = tf.argmax(logits, axis=-1)
    acc = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))

    # defining loss
    softmax_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    center_loss, centers, center_update_op = get_center_loss(features, labels)
    loss = softmax_loss + center_loss * LAMBDA

    # optimizer
    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    with tf.control_dependencies([center_update_op]):
        train_op = optimizer.minimize(loss, global_step)

    # loading data
    mnist = input_data.read_data_sets(DATASET_DIR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(NUM_EPOCHS):
            # training
            num_steps = TRAIN_SIZE // BATCH_SIZE
            train_acc = 0
            train_loss = 0
            for step in range(num_steps):

                x, y = mnist.train.next_batch(BATCH_SIZE)
                x = x.reshape([BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])
                _, train_acc_batch, train_loss_batch = sess.run([train_op, acc, loss], feed_dict={inputs: x, labels: y})
                train_acc += train_acc_batch
                train_loss += train_loss_batch
                sys.stdout.write("\r epoch %d, step %d, training accuracy %g, training loss %g" % (epoch + 1, step + 1, train_acc_batch, train_loss_batch))
                sys.stdout.flush()

            # testing
            x, y = mnist.test.next_batch(TEST_SIZE)
            x = x.reshape([TEST_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])
            test_acc, test_loss, output_features = sess.run([acc, loss, features], feed_dict={inputs: x, labels: y})
            print("\n epoch %d, testing accuracy %g, testing loss %g" % (epoch + 1, test_acc, test_loss))
            display_fig(output_features, y, epoch + 1)


if __name__ == '__main__':
    train()
