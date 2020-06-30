from keras import Model
from keras.layers import Dense, Activation
from googlenet.googlenet import create_googlenet
from keras.regularizers import l2
from keras.optimizers import Adam
from transfer_learning.data_loader import load_data
from tensorflow.keras.callbacks import TensorBoard
import time
import os

# tensorboard log file path
log_folder = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\logs"

# logfile name
log_name = "googlenet+SGD0.1+decay_1e-6+time{}".format(int(time.time()))

# pickle path
pickle_path = r"../../pickles"

# weight directory
weight_path = r"../../googlenet_weights.h5"

x_train, y_train, x_val, y_val = load_data(pickle_path)

tensorboard = TensorBoard(log_dir=os.path.join(log_folder, log_name))
print(os.path.join(log_folder, log_name))

base_model = create_googlenet(weight_path)

base_model.trainable = False

loss1_drop_fc = base_model.outputs[0]
loss1_classifier = Dense(5, kernel_regularizer=l2(0.0002))(loss1_drop_fc)
loss1_classifier_act = Activation('softmax')(loss1_classifier)

loss2_drop_fc = base_model.outputs[1]
loss2_classifier = Dense(5, kernel_regularizer=l2(0.0002))(loss2_drop_fc)
loss2_classifier_act = Activation('softmax')(loss2_classifier)

pool5_drop_7x7_s1 = base_model.outputs[2]
loss3_classifier = Dense(5, kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
loss3_classifier_act = Activation('softmax')(loss3_classifier)

model = Model(inputs=base_model.inputs, outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

adam = Adam(lr=0.001, decay=0.99)

model.compile(optimizer=adam,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


model.fit(x_train, [y_train, y_train, y_train],
          batch_size=64, epochs=200,
          validation_data=(x_val, [y_val, y_val, y_val]),
          callbacks=[tensorboard])