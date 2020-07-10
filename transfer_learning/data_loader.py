import pickle
import os

def load_data(pickle_path):

    x_path = "x.pickle"
    pickle_in = open(os.path.join(pickle_path, x_path), "rb")
    x = pickle.load(pickle_in)
    pickle_in.close()

    y_path = "y.pickle"
    pickle_in = open(os.path.join(pickle_path, y_path), "rb")
    y = pickle.load(pickle_in)
    pickle_in.close()

    return x, y