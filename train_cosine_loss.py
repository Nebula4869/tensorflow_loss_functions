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

MARGIN = 0.35
SCALE = 64

DATASET_DIR = './MNIST'
IMAGES_DIR = './images/cosine_loss'

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
            normalized_features = tf.nn.l2_normalize(net, axis=-1)
            feature_len = normalized_features.get_shape()[1]
            weights = tf.get_variable('fc3', [feature_len, NUM_CLASSES], tf.float32, slim.variance_scaling_initializer())
            normalized_weights = tf.nn.l2_normalize(weights, axis=0)
            logits = tf.matmul(normalized_features, normalized_weights)
        return logits, features


def get_cosine_loss(logits, labels):
    target_margin = tf.one_hot(labels, depth=NUM_CLASSES, on_value=MARGIN)
    target_logits = SCALE * (logits - target_margin)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=target_logits)


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
    axe.set_xlim(-2, 2)
    axe.set_ylim(-2, 2)
    axe.set_zlim(-2, 2)

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
    loss = get_cosine_loss(logits, labels)

    # optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss)

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
