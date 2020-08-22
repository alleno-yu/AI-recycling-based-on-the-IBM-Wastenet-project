from keras.utils import to_categorical
from data_preparation.data_loader import load_data
from transfer_learning.model_building import create_model
import os
import test_set.config as config

data = load_data()
x, y = data.load_test_data()

model = create_model()

model_path = os.path.join(config.model_path, "model.h5")
model.load_weights(model_path)

y = to_categorical(y)
history = model.evaluate(x, [y, y, y], batch_size=128, verbose=1)


print("loss: {:.2f}, final_classifier_loss: {:.2f}, final_classifier_acc: {:.2f}%".format(history[0], history[3], history[6]*100))