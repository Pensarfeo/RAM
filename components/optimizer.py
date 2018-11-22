import tensorflow as tf
import config
import pdb

class Optimizer():
    def __init__(self, classifier, labels_ph):
        self.listLogits = classifier.logits
        self.listSoftmax = classifier.softmax
        self.labels_ph = labels_ph
        # set Up Optimizer
        self.getLosses()
        self.getAccuracy()
        self.setTrainer()
    
    def getLosses(self):
        with tf.variable_scope('Losses'):

            # format
            logitsHistory = tf.convert_to_tensor(self.listLogits)
            logitsHistory = tf.transpose(logitsHistory, [1,0,2])

            # Fast Guess Reward!        
            scaleFastConvergeEntropy = [(lambda y: [(config.scaleFactor**y)/config.num_glimpses])(x) for x in range(0, config.num_glimpses)]
            scaleFastConvergeEntropy = tf.constant([scaleFastConvergeEntropy], dtype=tf.float32)
            scaleFastConvergeEntropy = tf.tile(scaleFastConvergeEntropy, [tf.shape(logitsHistory)[0], 1, 10])

            scaledFastConvergelogitsHistory = tf.multiply(scaleFastConvergeEntropy, logitsHistory)
            scaledFastConvergelogitsHistory = tf.reduce_mean(scaledFastConvergelogitsHistory, 1)

            batchFastConvergeEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scaledFastConvergelogitsHistory, labels = self.labels_ph)
            self.fastConvergeEntropy = tf.reduce_mean(batchFastConvergeEntropy)

            # Stable guess reward
            logitsHistory1 = tf.slice(logitsHistory, [0,0,0], [-1, tf.shape(logitsHistory)[1] - 1 ,-1])
            logitsHistory2 = tf.slice(logitsHistory, [0,1,0], [-1, tf.shape(logitsHistory)[1] - 1 ,-1])

            deltaLogitsHistory =  tf.abs(logitsHistory1 - logitsHistory2)


            deltaNGlipses = config.num_glimpses - 1
            
            scaleStableConvergeEntropy = [(lambda y: [((config.scaleFactor**(deltaNGlipses - y - 1))/deltaNGlipses)])(x) for x in range(0, deltaNGlipses)]
            scaleStableConvergeEntropy = tf.constant([scaleStableConvergeEntropy], dtype=tf.float32)
            scaleStableConvergeEntropy = tf.tile(scaleStableConvergeEntropy, [tf.shape(deltaLogitsHistory)[0], 1, 10])

            scaledStableConvergelogitsHistory = 1 - tf.multiply(scaleStableConvergeEntropy, deltaLogitsHistory)
            scaledStableConvergelogitsHistory = tf.reduce_mean(scaledStableConvergelogitsHistory, 1)

            batchStableConvergeEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scaledStableConvergelogitsHistory, labels = self.labels_ph)
            self.stableConvergeEntropy = tf.reduce_mean(batchStableConvergeEntropy)

            # Cross-entropy
            logits = self.listLogits[-1]
            entropy_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.labels_ph)
            self.entropy_value = tf.reduce_mean(entropy_value)
            
            # Hybric loss
            self.loss = self.entropy_value# + self.fastConvergeEntropy + self.stableConvergeEntropy
            self.var_list = tf.trainable_variables()
            self.grads = tf.gradients(self.loss, self.var_list)


    def getAccuracy(self):
        # Reward; caculated but not being used for any training...
        with tf.variable_scope('Losses'):
            softmax = tf.nn.softmax(self.listLogits[-1])
            labels_prediction = tf.argmax(softmax, 1)
            correct_predictions = tf.cast(tf.equal(labels_prediction, self.labels_ph), tf.float32)
            rewards = tf.expand_dims(correct_predictions, 1)
            rewards = tf.tile(rewards, (1, config.num_glimpses)) 
            self.accuracy = tf.reduce_mean(rewards)

    def setTrainer(self):
        with tf.variable_scope('Optimizer'):
            # Optimizer
            opt = tf.train.AdamOptimizer(0.0001)
            global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
            self.train_op = opt.apply_gradients(zip(self.grads, self.var_list), global_step=global_step)