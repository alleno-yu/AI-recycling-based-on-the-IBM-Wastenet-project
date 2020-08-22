from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures

from tensorflow import set_random_seed
from numpy.random import seed

import transfer_learning.config as config
from googlenet_folder.googlenet import create_googlenet


set_random_seed(2)
seed(2)


def create_model(weight_path=config.weight_path, unfreeze_list=config.unfreeze_list,
                 nb_classes=config.nb_classes, activation_layer=config.activation_layer,
                 loss=config.loss):

    # create googlenet
    base_model = create_googlenet(weight_path)

    # freeze some layers
    # for layer in base_model.layers[:]:
    #     if layer.name in unfreeze_list:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False

    # modify auxiliary layers
    loss1_drop_fc = base_model.outputs[0]
    # SVM1 = RandomFourierFeatures(output_dim=10, scale=1000, kernel_initializer="gaussian")(loss1_drop_fc)
    # loss1_classifier = Dense(nb_classes)(SVM1)
    loss1_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(loss1_drop_fc)
    loss1_classifier_act = Activation(activation_layer, name="aux_classifier-1")(loss1_classifier)

    # modify auxiliary layers
    loss2_drop_fc = base_model.outputs[1]
    # SVM2 = RandomFourierFeatures(output_dim=10, scale=1000, kernel_initializer="gaussian")(loss2_drop_fc)
    # loss2_classifier = Dense(nb_classes)(SVM2)
    loss2_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(loss2_drop_fc)
    loss2_classifier_act = Activation(activation_layer, name="aux_classifier-2")(loss2_classifier)

    # modify the final output layer
    pool5_drop_7x7_s1 = base_model.outputs[2]
    # SVM3 = RandomFourierFeatures(output_dim=10, scale=1000, kernel_initializer="gaussian")(pool5_drop_7x7_s1)
    # loss3_classifier = Dense(nb_classes)(SVM3)
    loss3_classifier = Dense(nb_classes, kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation(activation_layer, name='final_classifier')(loss3_classifier)

    # define inputs and outputs of the model
    model = Model(inputs=base_model.inputs,
                  outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

    # batch size=16
    adam = Adam(lr=2e-5)

    # compile model

    model.compile(optimizer=adam,
                  loss=[loss, loss, loss],
                  loss_weights=[0.3, 0.3, 1],
                  metrics=["accuracy"])

    print(model.summary())

    return model
