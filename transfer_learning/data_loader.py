import pickle
import os

def load_data(pickle_path):

    x_train_path = "x_train.pickle"
    pickle_in = open(os.path.join(pickle_path, x_train_path), "rb")
    x_train = pickle.load(pickle_in)
    pickle_in.close()

    y_train_path = "y_train.pickle"
    pickle_in = open(os.path.join(pickle_path, y_train_path), "rb")
    y_train = pickle.load(pickle_in)
    pickle_in.close()

    x_val_path = "x_val.pickle"
    pickle_in = open(os.path.join(pickle_path, x_val_path), "rb")
    x_val = pickle.load(pickle_in)
    pickle_in.close()

    y_val_path = "y_val.pickle"
    pickle_in = open(os.path.join(pickle_path, y_val_path), "rb")
    y_val = pickle.load(pickle_in)
    pickle_in.close()

    return x_train, y_train, x_val, y_val