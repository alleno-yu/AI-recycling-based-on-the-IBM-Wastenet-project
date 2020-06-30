import tensorflow as tf
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import gc
import imageio
folder_path = "dataset-resized"

# functions
def class_to_index(folder_path):
    class_dict = {}
    for count, value in enumerate(os.listdir(folder_path)):
        class_dict[value] = count
    return class_dict

def create_data(folder_path):
    training_data = []
    for folders in os.listdir(folder_path):
        for images in os.listdir(os.path.join(folder_path, folders)):
            class_num = class_dict[folders]
            # class_vector = [class_num, class_num, class_num]
            # print(class_vector)
            img = imageio.imread(os.path.join(folder_path, folders, images), pilmode='RGB')
            img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            img_array = img.transpose((2, 0, 1))

            # img_array = np.expand_dims(img, axis=0)
            training_data.append([img_array, class_num])
            print("processing image {}".format(images))
    return training_data

def normalized_data(training_data):
    x = []
    y = []
    for features, labels in training_data:
        x.append(features)
        y.append(labels)

    x = np.array(x)
    y = np.array(y)
    print(y.shape)
    print(x.shape)

    x = tf.keras.utils.normalize(x, axis=1)

    return x, y

def train_val_test(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.17, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.16, random_state=0)
    return x_train, y_train, x_val, y_val, x_test, y_test

def dump_pickle(x_train, y_train, x_val, y_val, x_test, y_test):
    pickle_out = open("x_train.pickle", "wb")
    pickle.dump(x_train, pickle_out, protocol=4)
    pickle_out.close()
    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out, protocol=4)
    pickle_out.close()

    pickle_out = open("x_val.pickle", "wb")
    pickle.dump(x_val, pickle_out, protocol=4)
    pickle_out.close()
    pickle_out = open("y_val.pickle", "wb")
    pickle.dump(y_val, pickle_out, protocol=4)
    pickle_out.close()

    pickle_out = open("x_test.pickle", "wb")
    pickle.dump(x_test, pickle_out, protocol=4)
    pickle_out.close()
    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out, protocol=4)
    pickle_out.close()

def image_display(image_array):
    plt.imshow(image_array)
    plt.show()


# Main part
class_dict = class_to_index(folder_path)
training_data = create_data(folder_path)
x, y = normalized_data(training_data)
del training_data
gc.collect()
x_train, y_train, x_val, y_val, x_test, y_test = train_val_test(x, y)
del x, y
gc.collect()
print(len(x_train), len(y_train), len(x_val), len(y_val), len(x_test), len(y_test))
dump_pickle(x_train, y_train, x_val, y_val, x_test, y_test)