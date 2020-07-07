import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import imageio
from sklearn.utils import shuffle

# put the image folder path here
dataset_path = r"../../datasets"

# put pickle folder path here
pickle_path = r"../../pickles"

# functions
def class_to_index(dataset_path):
    class_dict = {}
    for count, value in enumerate(os.listdir(dataset_path)):
        class_dict[value] = count
    return class_dict

def image_display(image_array):
    plt.imshow(image_array)
    plt.show()

# def create_data(folder_path):
#     training_data = []
#     for folders in os.listdir(folder_path):
#         for images in os.listdir(os.path.join(folder_path, folders)):
#             class_num = class_dict[folders]
#             img = imageio.imread(os.path.join(folder_path, folders, images), pilmode='RGB')
#             img = np.array(Image.fromarray(img).resize((224, 224))).astype("float32")
#
#             # this is for create batch shape (1, 224, 224, 3) from (224, 224, 3)
#             # img_array = np.expand_dims(img, axis=0)
#
#             img_array = img.transpose((2, 0, 1))
#             training_data.append([img_array, class_num])
#             print("processing image {}".format(images))
#     training_data = shuffle(training_data, random_state=20)
#     return training_data

def create_data(folder_path):
    training_data = []
    for folders in os.listdir(folder_path):
        for images in os.listdir(os.path.join(folder_path, folders)):
            class_num = class_dict[folders]
            img = imageio.imread(os.path.join(folder_path, folders, images), pilmode='RGB')
            img = np.array(Image.fromarray(img).resize((224, 224)))
            img_array = img.transpose((2, 0, 1))
            training_data.append([img_array, class_num])
            print("processing image {}".format(images))
    training_data = shuffle(training_data, random_state=20)
    return training_data

def normalized_data(training_data):
    x = []
    y = []
    for features, labels in training_data:
        x.append(features)
        y.append(labels)

    x = np.array(x)
    y = np.array(y)
    return x, y

def dump_pickle(x, y, pickle_path):

    x_path = "x.pickle"
    pickle_out = open(os.path.join(pickle_path, x_path), "wb")
    pickle.dump(x, pickle_out, protocol=4)
    pickle_out.close()

    y_path = "y.pickle"
    pickle_out = open(os.path.join(pickle_path, y_path), "wb")
    pickle.dump(y, pickle_out, protocol=4)
    pickle_out.close()

# Main part
class_dict = class_to_index(dataset_path)
training_data = create_data(dataset_path)
x, y = normalized_data(training_data)
dump_pickle(x, y, pickle_path)