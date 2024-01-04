import tensorflow as tf

def dice(targets, inputs):
    """
    Dice score is metric that is 1 for perfect precise model and 0 for no prediction and true values collision
    """
    if (tf.math.reduce_sum(targets) + tf.math.reduce_sum(inputs)) == 0:
        return 1.0
    return (2 * tf.math.reduce_sum(targets * inputs)) / (tf.math.reduce_sum(targets) + tf.math.reduce_sum(inputs))