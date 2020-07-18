import pickle
import os

import data_preparation.config as config

class load_data:

    def __init__(self, pickle_path=config.pickle_path):

        self.pickle_path = pickle_path
        self.x_train = "x_train.pickle"
        self.y_train = "y_train.pickle"
        self.x_test = "x_test.pickle"
        self.y_test = "y_test.pickle"

    def load_train_data(self):
        pickle_in = open(os.path.join(self.pickle_path, self.x_train), "rb")
        x_train = pickle.load(pickle_in)
        pickle_in.close()
        pickle_in = open(os.path.join(self.pickle_path, self.y_train), "rb")
        y_train = pickle.load(pickle_in)
        pickle_in.close()
        return x_train, y_train

    def load_test_data(self):
        pickle_in = open(os.path.join(self.pickle_path, self.x_test), "rb")
        x_test = pickle.load(pickle_in)
        pickle_in.close()
        pickle_in = open(os.path.join(self.pickle_path, self.y_test), "rb")
        y_test = pickle.load(pickle_in)
        pickle_in.close()
        return x_test, y_test