from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow import set_random_seed
from numpy.random import seed
from pathlib import Path
import numpy as np
import statistics
import os

from transfer_learning.model_building import create_model
from data_preparation.data_loader import load_data
from transfer_learning.learning_rate import step_decay
import transfer_learning.config as config

# base logging = "2", output error and fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed, both tensorflow and numpy
set_random_seed(2)
seed(2)

# load data and label
x_train, y_train, x_test, y_test = load_data(config.pickle_path)

# initialise empty val_acc and val_loss list
val_acc_list = []
val_loss_list = []
val_final_classifier_loss_list = []

# call cyclicLR
# clr = CyclicLR(base_lr=config.base_lr, max_lr=config.max_lr,
#                step_size=config.step_size)

# define callbacks
my_callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=config.patience_epochs)
    ,LearningRateScheduler(step_decay)
]

# call kFold
kfold = KFold(n_splits=config.nb_folds)

# initialise i
i = 0

# k fold loop
for train, val in kfold.split(x, y):

    #incremental i
    i = i+1

    # one hot encode labels
    y_train = to_categorical(y[train])
    y_val = to_categorical(y[val])

    # create model
    model = create_model()

    # fit model and copy to history
    history = model.fit(x[train], [y_train, y_train, y_train],
                        batch_size=config.batch_size, epochs=config.max_epochs,
                        validation_data=(x[val], [y_val, y_val, y_val]),
                        verbose=2, callbacks=my_callbacks)

    # define val_acc, val_loss
    val_acc = history.history["val_final_classifier_acc"]
    val_loss = history.history["val_loss"]
    val_final_classifier_loss = history.history["val_final_classifier_loss"]

    # append the extreme val_acc and val_loss to the list
    val_acc_list.append(max(val_acc))
    val_loss_list.append(min(val_loss))
    val_final_classifier_loss_list.append(min(val_final_classifier_loss))

    # create directory if not exist
    plt_directory = os.path.join(config.plt_root_folder, config.selective_folder)
    Path(plt_directory).mkdir(parents=True, exist_ok=True)

    # plot learning rate graph
    # h = clr.history
    # lr = h['lr']
    # plt.figure()
    # plt.plot(lr)
    # plt.xlabel("iterations")
    # plt.ylabel("learning rate")
    # plt.show()


    # plot figures for val_acc
    plt.figure()
    plt.plot(history.history['final_classifier_acc'])
    plt.plot(history.history['val_final_classifier_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(config.plt_root_folder, config.selective_folder, "accuracy{}.png".format(i)))

    # plot figures for val_loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(config.plt_root_folder, config.selective_folder, "loss{}.png".format(i)))

# print out results
val_acc_list = list(np.around(np.array(val_acc_list), 4))
print("validation accuracies are {}".format(val_acc_list))
print("the average validation accuracy is {:.2f}%".format(statistics.mean(val_acc_list)*100))

val_loss_list = list(np.around(np.array(val_loss_list), 2))
print("validation losses are {}".format(val_loss_list))
print("the average validation loss is {:.2f}".format(statistics.mean(val_loss_list)))

val_final_classifier_loss_list = list(np.around(np.array(val_final_classifier_loss_list), 2))
print("validation losses are {}".format(val_final_classifier_loss_list))
print("the average validation loss is {:.2f}".format(statistics.mean(val_final_classifier_loss_list)))