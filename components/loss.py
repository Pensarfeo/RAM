import tensorflow as tf
import config
import pdb

def losses(listLogits, labels_ph):
    with tf.variable_scope('Losses'):
        # format
        logitsHistory = tf.convert_to_tensor(listLogits)
        logitsHistory = tf.transpose(logitsHistory, [1,0,2])

        # Fast Guess Reward!        
        scaleFastConvergeEntropy = [(lambda y: [(config.scaleFactor**y)/config.num_glimpses])(x) for x in range(0, config.num_glimpses)]
        scaleFastConvergeEntropy = tf.constant([scaleFastConvergeEntropy], dtype=tf.float32)
        scaleFastConvergeEntropy = tf.tile(scaleFastConvergeEntropy, [tf.shape(logitsHistory)[0], 1, 10])

        scaledFastConvergelogitsHistory = tf.multiply(scaleFastConvergeEntropy, logitsHistory)
        scaledFastConvergelogitsHistory = tf.reduce_mean(scaledFastConvergelogitsHistory, 1)

        batchFastConvergeEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scaledFastConvergelogitsHistory, labels = labels_ph)
        fastConvergeEntropy = tf.reduce_mean(batchFastConvergeEntropy)

        # # Stable guess reward
        # logitsHistory1 = tf.slice(logitsHistory, [0,0,0], [-1, tf.shape(logitsHistory)[0] - 1 ,-1])
        # logitsHistory2 = tf.slice(logitsHistory, [0,1,0], [-1, tf.shape(logitsHistory)[0] - 1 ,-1])

        # deltaLogitsHistory = logitsHistory1 - logitsHistory2


        # deltaNGlipses = config.num_glimpses - 1
        # scaleStableConvergeEntropy = [(lambda y: [(config.scaleFactor**(y - deltaNGlipses)/deltaNGlipses)])(x) for x in range(0, deltaNGlipses)]
        # scaleStableConvergeEntropy = tf.constant([scaleStableConvergeEntropy], dtype=tf.float32)
        # scaleStableConvergeEntropy = tf.tile(scaleStableConvergeEntropy, [tf.shape(deltaLogitsHistory)[0], 1, 10])
        # pdb.set_trace()

        # scaledStableConvergelogitsHistory = tf.multiply(scaleStableConvergeEntropy, deltaLogitsHistory)
        # scaledStableConvergelogitsHistory = tf.reduce_mean(scaledStableConvergelogitsHistory, 1)

        # batchStableConvergeEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scaledStableConvergelogitsHistory, labels = labels_ph)
        # stableConvergeEntropy = tf.reduce_mean(batchStableConvergeEntropy)

        # Cross-entropy
        logits = listLogits[-1]
        entropy_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_ph)
        entropy_value = tf.reduce_mean(entropy_value)

        # Reward; caculated but not being used for any training...
        softmax = tf.nn.softmax(listLogits[-1])
        labels_prediction = tf.argmax(softmax, 1)
        correct_predictions = tf.cast(tf.equal(labels_prediction, labels_ph), tf.float32)
        rewards = tf.expand_dims(correct_predictions, 1)
        rewards = tf.tile(rewards, (1, config.num_glimpses)) 
        globalReward = tf.reduce_mean(rewards)
        
        # Hybric loss
        loss = entropy_value + fastConvergeEntropy #+ stableConvergeEntropy
        var_list = tf.trainable_variables()
        grads = tf.gradients(loss, var_list)

    return [grads, var_list, loss, globalReward, entropy_value, fastConvergeEntropy]