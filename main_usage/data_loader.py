import pickle
import os

from main_usage import config

# load test set pickles
# x_test is the image converted matrix, and y_test is the labels array

class load_data:

    def __init__(self, pickle_path=config.pickle_path):
        self.pickle_path = pickle_path
        self.x_test = "x_test.pickle"
        self.y_test = "y_test.pickle"

    def load_test_data(self):
        pickle_in = open(os.path.join(self.pickle_path, self.x_test), "rb")
        x_test = pickle.load(pickle_in)
        pickle_in.close()
        pickle_in = open(os.path.join(self.pickle_path, self.y_test), "rb")
        y_test = pickle.load(pickle_in)
        pickle_in.close()
        return x_test, y_test