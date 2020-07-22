# training parameters
batch_size = 64
patience_epochs = 20
max_epochs = 1000
nb_classes = 5
nb_folds = 5

# model settings
loss = "categorical_crossentropy"
activation_layer = "softmax"

# path config
# log_name = "googlenet"
# log_folder = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\logs"
plt_root_folder = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\plot_logs"
selective_folder = "Softmax"
pickle_path = r"../../pickles"
weight_path = r"../../googlenet_weights.h5"

# unfreeze layer config
unfreeze_list = ["inception_5b/5x5", "inception_5b/3x3", "inception_5b/1x1", "inception_5b/pool_proj",
                 "inception_5b/3x3_reduce", "inception_5b/5x5_reduce",
                 "inception_5a/1x1", "inception_5a/3x3", "inception_5a/5x5", "inception_5a/pool_proj",
                 "inception_5a/3x3_reduce", "inception_5a/5x5_reduce",
                 "inception_4e/1x1", "inception_4e/3x3", "inception_4e/5x5", "inception_4e/pool_proj",
                 "inception_4e/3x3_reduce", "inception_4e/5x5_reduce",
                 "inception_4d/pool_proj", "inception_4d/5x5", "inception_4d/3x3", "inception_4d/1x1"]
