from keras.models import load_model
from keras.utils import to_categorical
from data_preparation.data_loader import load_data
import transfer_learning.config as config
from transfer_learning.model_building import create_model

data = load_data()
x, y = data.load_test_data()

model = create_model()

model.load_weights(r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\model\model.h5")

y = to_categorical(y)
history = model.evaluate(x, [y, y, y], batch_size=128, verbose=1)

print("loss: {:.2f}, final_classifier_loss: {:.2f}, final_classifier_acc: {:.2f}%".format(history[0], history[3], history[6]*100))