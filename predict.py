import tensorflow as tf

from get_model import get_model


# Specifing model and simple prediction method
_model = get_model('active_model_weights.keras')
def predict(x = None):
    if x is None:
        raise Exception('Need image_path or image_tensor for prediction')
    if tf.is_tensor(x):
        return _model.predict(x[None])[0, ..., 1]
    if isinstance(x, str):
        return predict(
            tf.image.convert_image_dtype(
                tf.io.decode_jpeg(
                    tf.io.read_file(x)
                ),
                tf.float32)
        )


# test
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from private_settings import DATA_PATH
    img_path = DATA_PATH + 'train_v2/' + '91a6c1ee0.jpg'
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(
        tf.image.convert_image_dtype(
                tf.io.decode_jpeg(
                    tf.io.read_file(img_path)
                ),
                tf.float32)
    )
    ax2.imshow(predict(img_path))
    plt.show()