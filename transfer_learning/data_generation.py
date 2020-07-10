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
# convert folder name to label class name
def class_to_index(dataset_path):
    class_dict = {}
    for count, value in enumerate(os.listdir(dataset_path)):
        class_dict[value] = count
    return class_dict

# display images
def image_display(image_array):
    plt.imshow(image_array)
    plt.show()

# create training data
# imread(RGB) -> resize(224x224) -> transpose to channel first -> shuffle dataset
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

# split training data to data and label
def data_and_label(training_data):
    x = []
    y = []
    for features, labels in training_data:
        x.append(features)
        y.append(labels)

    x = np.array(x)
    y = np.array(y)
    return x, y

# pickle out data and label
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
x, y = data_and_label(training_data)
dump_pickle(x, y, pickle_path)