import tensorflow as tf
import config
import pdb

def losses(listLogits, labels_ph):
    with tf.variable_scope('Losses'):
        # Cross-entropy
        logits = listLogits[-1]
        entropy_value = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_ph)
        entropy_value = tf.reduce_mean(entropy_value)

        

        # This error is not well defined. A better erro can be:
        # p0*[alpha^-1*log(Sum{gamma^-1 *pi}/N) + alpha^-1*log(Sum{1 - gamma^(i + m *Detla(pi)}/M) + log(pn)]
        # Where N is the total number of loops and M = n -1, cause we do the delta!
        # alpha & gamma being 2 hyperparameters we select
        # IMPORTANT lop(pn) should be bigger than the other values
        # In summary, we reward initially good solutions, we don't punish initial uncertainty and reward final certainty!

        # Reward for 
        logitsHistory = tf.convert_to_tensor(listLogits)
        logitsHistory = tf.transpose(logitsHistory, [1,0,2])
        
        scaleFastConvergeEntropy = [(lambda y: [(config.scaleFactor**y)/config.num_glimpses])(x) for x in range(0, config.num_glimpses)]
        scaleFastConvergeEntropy = tf.constant(scaleFastConvergeEntropy, dtype=tf.float32)
        scaleFastConvergeEntropy = tf.tile(scaleFastConvergeEntropy, [1, 10])

        scaledFastConvergelogitsHistory = tf.map_fn(lambda lh: tf.multiply(lh, scaleFastConvergeEntropy), logitsHistory)
        scaledFastConvergelogitsHistory = tf.reshape(scaledFastConvergelogitsHistory, [tf.shape(labels_ph)[0]*config.num_glimpses, 10])

        fastConvergeLabelsList = tf.tile(labels_ph, [config.num_glimpses])

        batchFastConvergeEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scaledFastConvergelogitsHistory, labels = fastConvergeLabelsList)
        fastConvergeEntropy = tf.reduce_mean(batchFastConvergeEntropy)
        

        # entropyValueList = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logitsList, labels = labelsList)
        # entropyValueList = tf.reshape(entropyValueList, [config.num_glimpses, -1])

        # scaleList = [(lambda y: (config.scaleFactor**y))(x) for x in range(1,6)]
        # scaleList = tf.convert_to_tensor(scaleList, dtype=tf.float32)

        # entropyStepsDif = tf.reduce_mean(entropyValueList[0:-1], axis = 1)
        # entropyStepsDif = tf.multiply(scaleList, entropyStepsDif)
        # entropyStepsDif = tf.reduce_mean(entropyStepsDif)

        # Reward; caculated but not being used for any training...
        softmax = tf.nn.softmax(listLogits[-1])
        labels_prediction = tf.argmax(softmax, 1)
        correct_predictions = tf.cast(tf.equal(labels_prediction, labels_ph), tf.float32)
        rewards = tf.expand_dims(correct_predictions, 1)
        rewards = tf.tile(rewards, (1, config.num_glimpses)) 
        globalReward = tf.reduce_mean(rewards)

        # # Reward; punish jumping...
        # mu = tf.stack(retina.origin_coor_list)
        # sampled = tf.stack(retina.sample_coor_list)
        # gaussian = distributions.Normal(mu, config.loc_std)
        # _log = gaussian.log_prob(sampled)
        # _log = tf.reduce_sum(_log, 2)
        # _log = tf.transpose(_log)
        # _log_ratio = tf.reduce_mean(_log)
        # fastConvergeEntropy = tf.Print(fastConvergeEntropy, [fastConvergeEntropy])
        # Hybric loss
        loss = entropy_value #+ fastConvergeEntropy
        var_list = tf.trainable_variables()
        grads = tf.gradients(loss, var_list)
    return [grads, var_list, loss, globalReward]