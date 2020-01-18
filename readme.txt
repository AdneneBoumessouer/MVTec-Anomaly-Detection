TO DO:
- Add in train.py which object category to train on
- Test train.py on one class of MVTec Dataset
- Inversigate computed Residual Maps on existing loaded models

- Inversigate problem with loading a trained existing model: tf.keras.models.load_model(model_path) doesn't seem to work, see:
https://github.com/keras-team/keras/issues/10417
https://www.tensorflow.org/tutorials/keras/save_and_load

IDEAS: 
- Remove Parser and replace with a .json file for training setup
- Add initial epoch in ImageDataGenerator if training is resumed




OBSERVATIONS:

parsing works, training works




Retrieve local requirements:
pip freeze > requirements.txt
pip install -r requirements.txt 



Github:
url = git@github.com:AdneneBoumessouer/AutPr.git

Test: (to be deleted soon.)
