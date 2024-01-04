import tensorflow as tf
from tensorflow import keras

from public_settings import SEED


class Augment(tf.keras.layers.Layer):
    """
    Class for augmentation images. Due to reasons of preventing any bad infuence just flips and rotates image in all possible combinations.
    """
    def __init__(self, seed = SEED):
        super().__init__()

        #self.rng = np.random.default_rng(seed)
        self.rand_flip_imgs = keras.layers.RandomFlip(mode = "horizontal", seed = seed)
        self.rand_flip_masks = keras.layers.RandomFlip(mode = "horizontal", seed = seed)
    
    def call(self, imgs, masks):
        imgs = self.rand_flip_imgs(imgs)
        masks = self.rand_flip_masks(masks)

        #k = self.rng.choice(4)
        k = tf.random.categorical([[1/4] * 4], 1, dtype = tf.int32)[0, 0]
        imgs = tf.image.rot90(imgs, k)
        masks = tf.image.rot90(masks, k)

        #weights = tf.gather([0.3, 0.7], indices = tf.cast(masks[..., 1], tf.int32), name = 'cast_sample_weights')
        return imgs, masks#, weights