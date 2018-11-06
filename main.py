import os
import tensorlayer as tl
import tensorflow as tf
import numpy as np
from tensorflow.contrib import distributions

from data import Prepare_dataset
from lib.saveData import DataSaver
from lib.timer import Timer
import network

import config

# consts
EPHOCS = 2


# This function is defining the prob of going to an other point as the log prob of a gaussian.
# ence each jump, given the previous jump, is give by the previous jump origin (expectation of the gaussian)
# and a predifined sigma.
# This does not seem like a good procedure as the next step is not necessarely given information about the nature of the current object beeing seen.

# list of origin coordinates shape = [1, 320 (batchsize * M), 2 (x,y) ]
# list of sample coordinates shape = [1, 320 (batchsize * M), 2 (x,y) ]
# signa : a sigma for the gaussian... why?

if __name__ == '__main__':
    # Create placeholder
    images_ph = tf.placeholder(tf.float32, [None, 28 * 28])
    labels_ph = tf.placeholder(tf.int64, [None])

    # Create network
    logits, retina = network.setUp(images_ph)

    # Cross-entropy
    softmax = tf.nn.softmax(logits)
    entropy_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
    entropy_value = tf.reduce_mean(entropy_value)

    # Reward; caculated but not being used for any training...
    predict_label = tf.argmax(logits, 1)
    reward = tf.cast(tf.equal(predict_label, labels_ph), tf.float32)
    rewards = tf.expand_dims(reward, 1)
    rewards = tf.tile(rewards, (1, config.num_glimpses)) 
    reward = tf.reduce_mean(reward)

    # Reward; punish jumping...
    mu = tf.stack(retina.origin_coor_list)
    sampled = tf.stack(retina.sample_coor_list)
    gaussian = distributions.Normal(mu, config.loc_std)
    _log = gaussian.log_prob(sampled)
    _log = tf.reduce_sum(_log, 2)
    _log = tf.transpose(_log)
    _log_ratio = tf.reduce_mean(_log)

    # Hybric loss
    loss = entropy_value
    var_list = tf.trainable_variables()
    grads = tf.gradients(loss, var_list)

    # Optimizer
    opt = tf.train.AdamOptimizer(0.0001)
    global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
    train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)
    
    runName = "test"

    saver = tf.train.Saver()
    dataSaver = DataSaver('ephoch', 'iter', 'loss', 'reward', filename = './output/data/' + runName)

    # Train
    with tf.Session() as sess:
        # save gaph
        # tf.summary.FileWriter('./temp/graph').add_graph(sess.graph)

        mnist = Prepare_dataset(batch_size = config.batch_size)
        sess.run(tf.global_variables_initializer())

        timer = Timer(nsteps = (mnist.train_size // config.batch_size)*EPHOCS)
        
        for j in range(1, EPHOCS):
            for i in range(1, (mnist.train_size // config.batch_size)):
                # images, labels = mnist.train.next_batch(config.batch_size)
                images, labels = mnist()
                images = np.tile(images, [config.M, 1])
                labels = np.tile(labels, [config.M])

                _loss_value, _reward_value, _ = sess.run([loss, reward, train_op], feed_dict = {
                    images_ph: images,
                    labels_ph: labels
                })
                # print(i, i % 10)
                if i % (100) == 0:
                    dataSaver.add({
                        'ephoch': j
                        , 'iter': i
                        , 'loss': _loss_value
                        ,'reward': _reward_value
                    })
                    print(
                        'ephoc: ', j,
                        '\titer: ', i,
                        '\tloss: ', _loss_value,
                        # '\treward: ', _reward_value,
                        '\ttimeElapsed: ', timer.elapsed(step = (i + (j - 1) * (mnist.train_size // config.batch_size))),
                        '\tremaining: ', timer.left()
                    )

        modelSaveDir = 'output/trainedModels/' + runName
        if not os.path.exists(modelSaveDir):
            os.makedirs(modelSaveDir)
 
        saver.save(sess, modelSaveDir)