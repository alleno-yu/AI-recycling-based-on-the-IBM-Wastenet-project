from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imageio
import pickle
import os

import data_preparation.config as config


class process_data:

    def __init__(self, dataset_path=config.dataset_path, pickle_path=config.pickle_path,
                 test_size=config.test_size, random_state=config.random_state):

        self.dataset_path = dataset_path
        self.pickle_path = pickle_path
        self.test_size = test_size
        self.random_state = random_state
        self.class_dict = {}
        self.training_data = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.x = []
        self.y = []


    # create class dictionary
    def class_to_index(self):
        for count, value in enumerate(os.listdir(self.dataset_path)):
            self.class_dict[value] = count


    # create training data
    # imread(RGB) -> resize(224x224) -> transpose to channel first -> shuffle dataset
    def create_data(self):
        for folders in os.listdir(self.dataset_path):
            for images in os.listdir(os.path.join(self.dataset_path, folders)):
                class_num = self.class_dict[folders]
                img = imageio.imread(os.path.join(self.dataset_path, folders, images), pilmode='RGB')
                img = np.array(Image.fromarray(img).resize((224, 224)))
                img = img.transpose((2, 0, 1))
                self.training_data.append([img, class_num])
            print("processed class: {}".format(folders))


    # split data into training and test
    def train_and_test(self):
        for features, labels in self.training_data:
            self.x.append(features)
            self.y.append(labels)
        # self.x_train, self.x_test, self.y_train, self.y_test = \
        #     train_test_split(self.x, self.y, stratify=self.y, test_size=self.test_size, random_state=self.random_state)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size=self.test_size, random_state=self.random_state)
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)


    # dump training and test data into pickles
    def dump_pickle(self):
        pickle_out = open(os.path.join(self.pickle_path, "x_train.pickle"), "wb")
        pickle.dump(self.x_train, pickle_out, protocol=4)
        pickle_out.close()
        pickle_out = open(os.path.join(self.pickle_path, "x_test.pickle"), "wb")
        pickle.dump(self.x_test, pickle_out, protocol=4)
        pickle_out.close()
        pickle_out = open(os.path.join(self.pickle_path, "y_train.pickle"), "wb")
        pickle.dump(self.y_train, pickle_out, protocol=4)
        pickle_out.close()
        pickle_out = open(os.path.join(self.pickle_path, "y_test.pickle"), "wb")
        pickle.dump(self.y_test, pickle_out, protocol=4)
        pickle_out.close()
        print("dumped all pickles")


if __name__ == "__main__":
    TrashNet = process_data()
    TrashNet.class_to_index()
    TrashNet.create_data()
    TrashNet.train_and_test()
    TrashNet.dump_pickle()