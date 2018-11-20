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
EPHOCS = 25 * 5
TRAIN = True
runName = "feededGuess2LocationNet-rewardConfusionDecrease" 
modelSaveDir = os.path.join(os.getcwd(), 'output', runName, 'trainedModels')
modelSavePath = os.path.join(modelSaveDir, 'model.ckpt')
trainingString = 'training' if (TRAIN == True) else 'testing'
dataSavePath = os.path.join('output', runName, 'data', trainingString )
graphSavePath = os.path.join('output', runName, 'graph', trainingString )

print('graph location ======================================>')
print('tensorboard --logdir ', graphSavePath)
print('<====================================== graph location ')

if __name__ == '__main__':
    # Create placeholder
    images_ph = tf.placeholder(tf.float32, [None, 28 * 28])
    labels_ph = tf.placeholder(tf.int64, [None])

    # Create network
    classifier, retina = network.setUp(images_ph)

    with tf.variable_scope('Losses'):
        # Cross-entropy
        logits = classifier.logits[-1]
        entropy_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_ph)
        entropy_value = tf.reduce_mean(entropy_value)
        '''
            # This error is not well defined. A better erro can be:
            # p0*[alpha^-1*log(Sum{gamma^-1 *pi}/N) + alpha^-1*log(Sum{1 - gamma^(i + m *Detla(pi)}/M) + log(pn)]
            # Where N is the total number of loops and M = n -1, cause we do the delta!
            # alpha & gamma being 2 hyperparameters we select
            # IMPORTANT lop(pn) should be bigger than the other values
            # In summary, we reward initially good solutions, we don't punish initial uncertainty and reward final certainty!

    
            # Reward for 
            logitsList = tf.concat(classifier.logits, axis = 0)
            labelsList = tf.tile(labels_ph, [config.num_glimpses])
            entropyValueList = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logitsList, labels = labelsList)
            entropyValueList = tf.reshape(entropyValueList, [config.num_glimpses, -1])

            scaleList = [(lambda y: (config.scaleFactor**y))(x) for x in range(1,6)]
            scaleList = tf.convert_to_tensor(scaleList, dtype=tf.float32)

            entropyStepsDif = tf.reduce_mean(entropyValueList[0:-1], axis = 1)
            entropyStepsDif = tf.multiply(scaleList, entropyStepsDif)
            entropyStepsDif = tf.reduce_mean(entropyStepsDif)

            # Reward; caculated but not being used for any training...
            softmax = tf.nn.softmax(classifier.logits[-1])
            labels_prediction = tf.argmax(softmax, 1)
            correct_predictions = tf.cast(tf.equal(labels_prediction, labels_ph), tf.float32)
            rewards = tf.expand_dims(correct_predictions, 1)
            rewards = tf.tile(rewards, (1, config.num_glimpses)) 
            reward = tf.reduce_mean(rewards)
        '''     
        # Reward; punish jumping...
        # mu = tf.stack(retina.origin_coor_list)
        # sampled = tf.stack(retina.sample_coor_list)
        # gaussian = distributions.Normal(mu, config.loc_std)
        # _log = gaussian.log_prob(sampled)
        # _log = tf.reduce_sum(_log, 2)
        # _log = tf.transpose(_log)
        # _log_ratio = tf.reduce_mean(_log)
        
        # Hybric loss
        loss = entropy_value
        var_list = tf.trainable_variables()
        grads = tf.gradients(loss, var_list)

    with tf.variable_scope('Optimizer'):
        # Optimizer
        opt = tf.train.AdamOptimizer(0.0001)
        global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)
    

    saver = tf.train.Saver()
    
    if TRAIN:
        
        dataSaver = DataSaver('ephoch', 'iter', 'loss', 'reward', filename = dataSavePath)

        # Train
        with tf.Session() as sess:
            # save gaph
            tf.summary.FileWriter(graphSavePath).add_graph(sess.graph)

            mnist = Prepare_dataset(batch_size = config.batch_size)
            tf.global_variables_initializer().run()


            if os.path.isfile(modelSavePath + ".index"):
                saver.restore(sess, modelSavePath)

            timer = Timer(nsteps = (mnist.train_size // config.batch_size)*EPHOCS)

            if not os.path.exists(modelSaveDir):
                os.makedirs(modelSaveDir)

            for j in range(0, EPHOCS):
                for i in range(1, (mnist.train_size // config.batch_size)):
                    # images, labels = mnist.train.next_batch(config.batch_size)
                    images, labels = mnist(epoch = j)
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
                            '\treward: ', _reward_value,
                            '\ttimeElapsed: ', timer.elapsed(step = (i + j * (mnist.train_size // config.batch_size))),
                            '\tremaining: ', timer.left()
                        )

                if i % (25) == 0:
                    saver.save(sess, modelSavePath)

    
            

    else:
        # --------------------------------------------------------------
        # test loop
        # --------------------------------------------------------------

        dataSaver = DataSaver('n', 'softmax', 'label', filename = dataSavePath, divider=',')
        
        with tf.Session() as sess:
            # save gaph
            # tf.summary.FileWriter('./temp/graph').add_graph(sess.graph)

            trainingBatchSize = config.batch_size
            mnist = Prepare_dataset(batch_size = trainingBatchSize)
            tf.global_variables_initializer().run()
            saver.restore(sess, modelSavePath)

            for i in range(1, (mnist.train_size // trainingBatchSize)):
                images, labels = mnist()

                _softmax = sess.run([softmax], feed_dict = {
                    images_ph: images,
                    labels_ph: labels
                })

                for j in range(0, trainingBatchSize):
                    dataSaver.add({
                            'n': i + j
                            ,'softmax': _softmax[0][j]
                            ,'label': labels[j]
                    })
                # print(i, i % 10)
                if i % (1000 // trainingBatchSize) == 0:
                    # dataSaver.add({
                    #     'n': i
                    #     ,'prediction': _labels_prediction
                    #     ,'label': labels
                    # })
                    print(
                        '\\n: ', i
                        # , '\tloss: ', _labels_prediction
                        # , '\treward: ', _reward_value
                    )