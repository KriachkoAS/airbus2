from tensorflow import keras

from datasets import ds_filtered_train, ds_valid
from get_model import get_model
from augment import Augment
from metrics import dice


def train(weights_path = None, ds_train = ds_filtered_train, ds_valid = ds_valid,
        batch_size = 1,
        optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [dice],
        epochs = 1,
        ):
    """
    Fuction for using the same way that got weights trained
    """

    # Prepare train data
    train_batches = (
        ds_train
        .repeat()
        .map(Augment())
        .batch(batch_size))

    # Load and compile model
    model = get_model(weights_path)
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrics,
    )
    # Fine tuning
    return model, model.fit(
            train_batches,
            epochs = epochs,
            steps_per_epoch = len(ds_train),
            validation_data = ds_valid.batch(batch_size),
            callbacks = [keras.callbacks.ModelCheckpoint(
                filepath = 'checkpoints/',
                monitor = 'val_mean-IoU',
                mode = 'max',
                save_best_only = False,
                )],
            )


# test
if __name__ == '__main__':
    from train_splitted_dfs import df_filtered_mask
    import tensorflow as tf
    from datasets import _get_ds_item
    ds_train = tf.data.Dataset.from_tensor_slices(df_filtered_mask.loc['train'].iloc[:20]).map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)
    ds_valid = tf.data.Dataset.from_tensor_slices(df_filtered_mask.loc['valid'].iloc[:10]).map(_get_ds_item, num_parallel_calls = tf.data.AUTOTUNE)
    train(weights_path = 'active_model_weights.keras', ds_train = ds_train, ds_valid = ds_valid)