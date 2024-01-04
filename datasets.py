import tensorflow as tf

from train_splitted_dfs import df_mask, df_filtered_mask

from public_settings import IMG_SIZE
from private_settings import DATA_PATH


# Generates datasets from respective dataframes that consist of summary encoded pixels
ds_filtered_train = tf.data.Dataset.from_tensor_slices(df_filtered_mask.loc['train'])
ds_filtered_valid = tf.data.Dataset.from_tensor_slices(df_filtered_mask.loc['valid'])
ds_filtered_test = tf.data.Dataset.from_tensor_slices(df_filtered_mask.loc['test'])
ds_train = tf.data.Dataset.from_tensor_slices(df_mask.loc['train'])
ds_valid = tf.data.Dataset.from_tensor_slices(df_mask.loc['valid'])
ds_test = tf.data.Dataset.from_tensor_slices(df_mask.loc['test'])

# Method for generating binary masks using only summery encoded pixels
def _decode_rle(rle, shape = (768, 768)):
    shape = tf.convert_to_tensor(shape, tf.int64)
    rle = tf.strings.to_number(tf.strings.split(rle), tf.int64)
    starts = rle[::2] - 1
    lens = rle[1::2]
    ones_len = tf.reduce_sum(lens)
    ones = tf.ones([ones_len], tf.uint8)
    # Make scattering indices
    r = tf.range(ones_len)
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, 'right')
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)
    return tf.transpose(tf.reshape(
        tf.scatter_nd(tf.expand_dims(idx, 1), ones, [tf.math.reduce_prod(shape)]),
        shape))

# Method for getting image tensor and one-hot encoded overall mask
def _get_ds_item(tensor):
    img_id = tensor[0]
    encoded_pixels = tensor[1]
    img = tf.image.convert_image_dtype(
        tf.io.decode_jpeg(
            tf.io.read_file(DATA_PATH + 'train_v2/' + img_id)
        ),
        tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    mask = _decode_rle(encoded_pixels)
    mask = tf.image.resize(mask[..., None], IMG_SIZE, method='nearest')[..., 0]
    #weights = tf.gather([0.05, 0.95], indices = tf.cast(mask, tf.int32), name = 'cast_sample_weights')
    mask = tf.cast(tf.one_hot(mask, 2), dtype=tf.float32)
    return img, mask

# Applying previous method
ds_filtered_train = ds_filtered_train.map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)
ds_filtered_valid = ds_filtered_valid.map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)
ds_filtered_test = ds_filtered_test.map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)
ds_train = ds_train.map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)
ds_valid = ds_valid.map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)
ds_test = ds_test.map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)


# test
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    for img, mask in ds_filtered_train.take(1):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax2.imshow(mask[..., 1])
        plt.show()