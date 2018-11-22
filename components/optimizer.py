import tensorflow as tf
import config

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
            
            labels = tf.stop_gradient(self.labels_ph)
            labelsOneHot = tf.one_hot(labels, 10)

            # format
            softmaxHistory = tf.convert_to_tensor(self.listSoftmax)
            softmaxTotNum = tf.cast(tf.shape(softmaxHistory)[0], tf.float32)
            softmaxHistory = tf.transpose(softmaxHistory, [1,0,2])

            # Fast Guess Reward!        
            scaleFastConvergeEntropy = [(lambda y: [(config.scaleFactor**y)/config.num_glimpses])(x) for x in range(0, config.num_glimpses)]
            scaleFastConvergeEntropy = tf.constant([scaleFastConvergeEntropy], dtype=tf.float32)
            scaleFastConvergeEntropy = tf.tile(scaleFastConvergeEntropy, [tf.shape(softmaxHistory)[0], 1, 10])

            softmaxFastConvergeHistory = tf.multiply(scaleFastConvergeEntropy, softmaxHistory)
            softmaxFastConvergeAverage = tf.reduce_mean(softmaxFastConvergeHistory, 1)

            batchFastConvergeEntropy = tf.multiply(softmaxFastConvergeAverage, labelsOneHot)
            batchFastConvergeEntropy = tf.reduce_sum(batchFastConvergeEntropy, 1)
            batchFastConvergeEntropy = tf.log(batchFastConvergeEntropy)
            self.fastConvergeEntropy = tf.reduce_mean(batchFastConvergeEntropy) * -1

            # Stable guess reward
            softmaxHistory1 = tf.slice(softmaxHistory, [0,0,0], [-1, tf.shape(softmaxHistory)[1] - 1 ,-1])
            softmaxHistory2 = tf.slice(softmaxHistory, [0,1,0], [-1, tf.shape(softmaxHistory)[1] - 1 ,-1])
            softmaxHistoryD =  tf.abs(softmaxHistory1 - softmaxHistory2)

            deltaNGlipses = config.num_glimpses - 1            
            scaleStableConvergeEntropy = [(lambda y: [((config.scaleFactor**(deltaNGlipses - y - 1))/deltaNGlipses)])(x) for x in range(0, deltaNGlipses)]
            scaleStableConvergeEntropy = tf.constant([scaleStableConvergeEntropy], dtype=tf.float32)
            scaleStableConvergeEntropy = tf.tile(scaleStableConvergeEntropy, [tf.shape(softmaxHistory)[0], 1, 10])

            softmaxHistoryD = tf.multiply(scaleStableConvergeEntropy, softmaxHistoryD)
            softmaxHistoryD = tf.reduce_mean(softmaxHistory, 1)
            softmaxHistoryD = 1 - softmaxHistoryD

            batchStableConvergeEntropy = tf.multiply(softmaxHistoryD, labelsOneHot)
            batchStableConvergeEntropy = tf.reduce_sum(batchStableConvergeEntropy, 1)
            batchStableConvergeEntropy = tf.log(batchStableConvergeEntropy)
            
            self.stableConvergeEntropy = tf.reduce_mean(batchStableConvergeEntropy) * -1         

            # Cross-entropy
            softmax = self.listSoftmax[-1]
            entropy_value = tf.multiply(softmax, labelsOneHot)
            entropy_value = tf.reduce_sum(entropy_value, 1)
            entropy_value = tf.log(entropy_value)
            self.entropy_value = tf.reduce_mean(entropy_value) * -1
            
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