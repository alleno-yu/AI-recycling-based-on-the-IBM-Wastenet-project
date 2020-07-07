from keras import Model
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, Adadelta
from keras.utils import to_categorical

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from numpy.random import seed
import numpy as np
from pathlib import Path
import statistics
import os

from transfer_learning.data_loader import load_data
from googlenet.googlenet import create_googlenet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

set_random_seed(2)
seed(1)

# parameters selection
batch_size = 128
patience_epochs = 20
max_epochs = 1000
learning_rate = 0.001
nb_classes = 5
nb_folds = 5
decay = 0
loss = "sparse_categorical_crossentropy"
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

# x, y = load_data(pickle_path)
x, y = load_data(pickle_path)
kfold = StratifiedKFold(n_splits=nb_folds)

i = 1
val_acc_list = []
val_loss_list = []

for train, val in kfold.split(x, y):

    base_model = create_googlenet(weight_path)

    for layer in base_model.layers[:]:
        if layer.name in unfreeze_list:
            layer.trainable = False
        else:
            layer.trainable = False

    loss1_drop_fc = base_model.outputs[0]
    loss1_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(loss1_drop_fc)
    loss1_classifier_act = Activation(activation_layer)(loss1_classifier)

    loss2_drop_fc = base_model.outputs[1]
    loss2_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(loss2_drop_fc)
    loss2_classifier_act = Activation(activation_layer)(loss2_classifier)

    pool5_drop_7x7_s1 = base_model.outputs[2]
    loss3_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation(activation_layer, name='final_classifier')(loss3_classifier)

    model = Model(inputs=base_model.inputs, outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

    adam = Adam(lr=learning_rate, decay=decay)
    # adadelta = Adadelta(lr=lr, decay=decay)

    model.compile(optimizer=adam,
                  loss=loss,
                  metrics=["accuracy"])

    # y_train = to_categorical(y[train])
    # y_val = to_categorical(y[val])

    # print(model.summary())

    my_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_epochs),
        TensorBoard(log_dir=os.path.join(log_folder, log_name)),
    ]

    history = model.fit(x[train], [y[train], y[train], y[train]],
                        batch_size=batch_size, epochs=max_epochs,
                        validation_data=(x[val], [y[val], y[val], y[val]]),
                        verbose=2,
                        callbacks=my_callbacks)

    # print(history.history.keys())

    val_acc = history.history["val_final_classifier_acc"]
    # print("the highest validation accuracy is {:.2f}% at epoch {}".format(max(val_acc)*100, val_acc.index(max(val_acc))+1))
    val_loss = history.history["val_loss"]
    # print("the lowest validation loss is {:.2f} at epoch {}".format(min(val_loss), val_loss.index(min(val_loss))+1))

    val_acc_list.append(max(val_acc))
    val_loss_list.append(min(val_loss))

    # create directory if not exist
    plt_directory = os.path.join(plt_root_folder, selective_folder)
    Path(plt_directory).mkdir(parents=True, exist_ok=True)

    # # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['final_classifier_acc'])
    plt.plot(history.history['val_final_classifier_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plt_root_folder, selective_folder, "accuracy{}.png".format(i)))

    # # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plt_root_folder, selective_folder, "loss{}.png".format(i)))

    i = i+1


val_acc_list = list(np.around(np.array(val_acc_list), 2))
print("validation accuracies are {}".format(val_acc_list))
print("the average validation accuracy is {:.2f}%".format(statistics.mean(val_acc_list)*100))


val_loss_list = list(np.around(np.array(val_loss_list), 2))
print("validation losses are {}".format(val_loss_list))
print("the average validation loss is {:.2f}".format(statistics.mean(val_loss_list)))






