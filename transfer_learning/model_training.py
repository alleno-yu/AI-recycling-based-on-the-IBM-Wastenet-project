from keras import Model
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, Adadelta
from keras.utils import to_categorical

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from numpy.random import seed
import numpy as np
from pathlib import Path
import statistics
import os

from transfer_learning.data_loader import load_data
from googlenet.googlenet import create_googlenet

# base logging = "2", output error and fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed, both tensorflow and numpy
set_random_seed(2)
seed(2)

# parameters selection
batch_size = 128
patience_epochs = 20
max_epochs = 1
learning_rate = 0.001
nb_classes = 5
nb_folds = 5
decay = 0
loss = "categorical_crossentropy"
activation_layer = "softmax"

# path
log_name = "googlenet+lr_{}".format(learning_rate)
log_folder = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\logs"
plt_root_folder = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\plot_logs"
selective_folder = "lr{}+Softmax".format(learning_rate)
pickle_path = r"../../pickles"
weight_path = r"../../googlenet_weights.h5"

# unfreeze layer list
unfreeze_list = ["loss1/fc", "loss2/fc"]

# load data and label
x, y = load_data(pickle_path)

# call k-fold cross validation function
kfold = StratifiedKFold(n_splits=nb_folds)
# kfold = KFold(n_splits=nb_folds)

# initialise i
i = 0


# define val_acc and val_loss list for k-fold average calculation
val_acc_list = []
val_loss_list = []

# k fold loop
for train, val in kfold.split(x, y):

    # i incremental
    i = i+1

    # create googlenet
    base_model = create_googlenet(weight_path)

    # freeze some layers
    for layer in base_model.layers[:]:
        if layer.name in unfreeze_list:
            layer.trainable = False
        else:
            layer.trainable = False

    # modify auxiliary layers
    loss1_drop_fc = base_model.outputs[0]
    loss1_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(loss1_drop_fc)
    loss1_classifier_act = Activation(activation_layer)(loss1_classifier)

    # modify auxiliary layers
    loss2_drop_fc = base_model.outputs[1]
    loss2_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(loss2_drop_fc)
    loss2_classifier_act = Activation(activation_layer)(loss2_classifier)

    # modify the final output layer
    pool5_drop_7x7_s1 = base_model.outputs[2]
    loss3_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation(activation_layer, name='final_classifier')(loss3_classifier)

    # define inputs and outputs of the model
    model = Model(inputs=base_model.inputs,
                  outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

    # set adam optimizer
    adam = Adam(lr=learning_rate, decay=decay)
    # adadelta = Adadelta(lr=learning_rate, decay=decay)

    # compile model
    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=["accuracy"])

    # one hot encode labels
    y_train = to_categorical(y[train])
    y_val = to_categorical(y[val])

    # y_train = y[train]
    # y_val = y[val]

    # print(model.summary())

    # define callbacks
    my_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_epochs),
        TensorBoard(log_dir=os.path.join(log_folder, log_name)),
    ]

    # fit model and copy to history
    history = model.fit(x[train], [y_train, y_train, y_train],
                        batch_size=batch_size, epochs=max_epochs,
                        validation_data=(x[val], [y_val, y_val, y_val]),
                        verbose=2, callbacks=my_callbacks)

    # print(history.history.keys())

    # define val_acc, val_loss
    val_acc = history.history["val_final_classifier_acc"]
    val_loss = history.history["val_loss"]

    # append the extreme val_acc and val_loss to the list
    val_acc_list.append(max(val_acc))
    val_loss_list.append(min(val_loss))

    # create directory if not exist
    plt_directory = os.path.join(plt_root_folder, selective_folder)
    Path(plt_directory).mkdir(parents=True, exist_ok=True)

    # plot figures for val_acc
    plt.figure()
    plt.plot(history.history['final_classifier_acc'])
    plt.plot(history.history['val_final_classifier_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plt_root_folder, selective_folder, "accuracy{}.png".format(i)))

    # plot figures for val_loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plt_root_folder, selective_folder, "loss{}.png".format(i)))



# print out results
val_acc_list = list(np.around(np.array(val_acc_list), 4))
print("validation accuracies are {}".format(val_acc_list))
print("the average validation accuracy is {:.2f}%".format(statistics.mean(val_acc_list)*100))

val_loss_list = list(np.around(np.array(val_loss_list), 2))
print("validation losses are {}".format(val_loss_list))
print("the average validation loss is {:.2f}".format(statistics.mean(val_loss_list)))






