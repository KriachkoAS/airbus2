## Airbus ship semantic segmentation ##

For correct working of module private_settings.py need to be generated where variable `DATA_PATH` should be specified and ended with `/`.

Due to comments of https://www.kaggle.com/c/airbus-ship-detection/data competitors
resnet based unet was choosen, I used the one found at https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/resnet50_unet.py.
Project got just basic set of tools to deal with model training and prediction with minimum settings required. The only note that there is no ready script to turn
checkpoints into .keras files due to model was learned at another device where histori was lost to deal with. Current active_model_weights.keras file serves as
trained model weights source (10 hours,  5 epochs).

For using model prediction importing `predict` from `predict` module and use it with file path specified.
For generating checkpoints and getting trained model use `train` function from `train_model` module. Settings can be ommited due to default specified.
exploratory.ipynb available for demonstrating available dataset.
requirements.txt specified for python environment settings.