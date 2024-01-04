import pandas as pd

from private_settings import DATA_PATH
from public_settings import CORRUPTED_TRAIN_IMAGE_IDS


# df_mask is DataFrame for summed encoded pixels splitted by train, valid, test examples while df_filtered mask is the same without empy mask 
df_mask = pd.read_csv(DATA_PATH + 'train_ship_segmentations_v2.csv', dtype = 'string', index_col = 'ImageId')
df_mask = pd.DataFrame((df_mask['EncodedPixels'].fillna('') + ' ').groupby('ImageId').sum().str[:-1]).drop(CORRUPTED_TRAIN_IMAGE_IDS).reset_index()
df_mask = df_mask.sample(len(df_mask), random_state = 42)
_index = ['train'] * int(len(df_mask) * 0.8) + ['valid'] * int(len(df_mask) * 0.1)
_index += ['test'] * (len(df_mask) - len(_index))
df_mask.index = _index
df_filtered_mask = df_mask[df_mask['EncodedPixels'] != '']


# demonstrate
if __name__ == '__main__':
    print(df_mask)
    print(df_filtered_mask)