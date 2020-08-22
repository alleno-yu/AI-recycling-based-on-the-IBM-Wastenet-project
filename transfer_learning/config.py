# training parameters
batch_size = 16
patience_epochs = 100
max_epochs = 1000
nb_classes = 5
nb_folds = 10
rs = 99


# model settings
loss = "categorical_crossentropy"
activation_layer = "softmax"
# loss = "hinge"
# activation_layer = "linear"


# path config

csv_path = r"C:\Users\allen\Desktop\Final_Project\IBM_Wastenet\logs\training.txt"
model_path = r"../../model"
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

                 "loss2/conv", "loss2/fc",

                 "inception_4d/5x5", "inception_4d/3x3", "inception_4d/1x1", "inception_4d/pool_proj",
                 "inception_4d/5x5_reduce", "inception_4d/3x3_reduce",

                 "inception_4c/1x1", "inception_4c/3x3", "inception_4c/5x5", "inception_4c/pool_proj",
                 "inception_4c/3x3_reduce", "inception_4c/5x5_reduce",
                 
                 "inception_4b/1x1", "inception_4b/3x3", "inception_4b/5x5", "inception_4b/pool_proj",
                 "inception_4b/3x3_reduce", "inception_4b/5x5_reduce",

                 "loss1/conv", "loss1/fc",

                 "inception_4a/1x1", "inception_4a/3x3", "inception_4a/5x5", "inception_4a/pool_proj",
                 "inception_4a/3x3_reduce", "inception_4a/5x5_reduce",

                 "inception_3b/1x1", "inception_3b/3x3", "inception_3b/5x5", "inception_3b/pool_proj",
                 "inception_3b/3x3_reduce", "inception_3b/5x5_reduce",
                 
                 "inception_3a/1x1", "inception_3a/3x3", "inception_3a/5x5", "inception_3a/pool_proj",
                 "inception_3a/3x3_reduce", "inception_3a/5x5_reduce",

                 "conv2/3x3", "conv2/3x3_reduce", "conv1/7x7_s2"
                 ]
