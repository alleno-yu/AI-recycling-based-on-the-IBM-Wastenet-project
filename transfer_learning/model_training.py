from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow import set_random_seed
from numpy.random import seed
from pathlib import Path
import numpy as np
import os

from transfer_learning.model_building import create_model
from data_preparation.data_loader import load_data
import transfer_learning.config as config

# from transfer_learning.learning_rate import step_decay
# from cyclic_lr.clr_callback import CyclicLR

# base logging = "2", output error and fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed, both tensorflow and numpy
set_random_seed(2)
seed(2)

# load training data and label
data = load_data()
x, y = data.load_train_data()

# initialise empty val_acc and val_loss list
val_acc_list = []
val_loss_list = []
val_final_classifier_loss_list = []

# define the checkpoint
# model_path = os.path.join(config.model_path, "model.h5")
early_stopping = EarlyStopping(monitor='val_final_classifier_acc', mode='max',
                               min_delta=0.001, verbose=2, patience=config.patience_epochs)

# datagen = ImageDataGenerator(
#     # width_shift_range=0.0,
#     # height_shift_range=0.0,
#     # brightness_range=None,
#     # shear_range=0.0,
#     # zoom_range=0.0,
#     horizontal_flip=True,
#     vertical_flip=True,
#     rotation_range=15,
#
# )


# checkpoint = ModelCheckpoint(model_path, monitor='val_loss',
#                              verbose=1, save_weights_only=True, save_best_only=True, mode='min')
# clr = CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=375, mode='triangular2')
# lr_scheduler = LearningRateScheduler(step_decay)
# csv_logger = CSVLogger(filename=config.csv_path, append=True)

# define callbacks
my_callbacks = [
    early_stopping
    # csv_logger
    # checkpoint
]

# call kFold
kfold = KFold(n_splits=config.nb_folds, shuffle=True, random_state=config.rs)

# initilise i
i = 0

# k fold loop
for train, val in kfold.split(x, y):

    # one hot encode labels
    x_train = x[train]
    x_val = x[val]
    y_train = to_categorical(y[train])
    y_val = to_categorical(y[val])

    # incremental
    i = i+1

    # fit to x_train
    # datagen.fit(x_train)


    # def generate_data_generator(datagen, x_train, y_train):
    #     genX1 = datagen.flow(x_train, y_train, batch_size=16)
    #     while True:
    #         X1i = genX1.next()
    #         yield X1i[0], [X1i[1], X1i[1], X1i[1]]


    # create model
    model = create_model()

    # fit model and copy to history
    # history = model.fit_generator(generator=generate_data_generator(datagen, x_train, y_train),
    #                               steps_per_epoch=len(x_train)/16,
    #                               validation_data=(x_val, [y_val, y_val, y_val]),
    #                               verbose=2, callbacks=my_callbacks, epochs=config.max_epochs)

    history = model.fit(x=x_train, y=[y_train, y_train, y_train],
                        batch_size=config.batch_size, epochs=config.max_epochs,
                        validation_data=(x_val, [y_val, y_val, y_val]),
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
    # plt_directory = os.path.join(config.plt_root_folder, config.selective_folder)
    # Path(plt_directory).mkdir(parents=True, exist_ok=True)

    # plot learning rate graph
    # h = clr.history
    # lr = h['lr']
    # plt.figure()
    # plt.plot(lr)
    # plt.xlabel("iterations")
    # plt.ylabel("learning rate")
    # plt.show()

    # print out results
    val_acc_list = list(np.around(np.array(val_acc_list), 4))
    print("validation accuracies are {}".format(val_acc_list))
    val_acc_mean = sum(val_acc_list) / len(val_acc_list)
    print("the average validation accuracy is {:.2f}%".format(val_acc_mean * 100))

    val_loss_list = list(np.around(np.array(val_loss_list), 3))
    print("total validation losses are {}".format(val_loss_list))
    val_loss_mean = sum(val_loss_list) / len(val_loss_list)
    print("the average total validation loss is {:.3f}".format(val_loss_mean))

    val_final_classifier_loss_list = list(np.around(np.array(val_final_classifier_loss_list), 3))
    print("final layer validation losses are {}".format(val_final_classifier_loss_list))
    val_final_loss_mean = sum(val_final_classifier_loss_list) / len(val_final_classifier_loss_list)
    print("the average final layer validation loss is {:.3f}".format(val_final_loss_mean))