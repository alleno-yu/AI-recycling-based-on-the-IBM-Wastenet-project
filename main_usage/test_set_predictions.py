from keras.utils import to_categorical
from googlenet_model.googlenet import create_googlenet
from main_usage.data_loader import load_data
from main_usage.config import weights_path
from tensorflow.keras.optimizers import Adam

from tensorflow import set_random_seed
from numpy.random import seed

set_random_seed(2)
seed(2)

# load the test set data
data = load_data()
x, y = data.load_test_data()

# build and compile the model
model = create_googlenet(weights_path)
model.compile(optimizer=Adam(2e-5),
              loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"],
              loss_weights=[0.3, 0.3, 1],
              metrics=["accuracy"])

y = to_categorical(y)

# evaluate the test set
history = model.evaluate(x, [y, y, y], batch_size=128, verbose=1)