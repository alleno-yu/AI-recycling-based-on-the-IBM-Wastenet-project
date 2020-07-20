# training parameters
batch_size = 32
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
# unfreeze_list = []
# unfreeze_list = ["loss1/fc", "loss2/fc"]
# unfreeze_list = ["inception_5b/5x5", "inception_5b/3x3", "inception_5b/1x1", "inception_5b/pool_proj"]
unfreeze_list = ["inception_5b/5x5", "inception_5b/3x3", "inception_5b/1x1", "inception_5b/pool_proj",
                 "loss1/fc", "loss2/fc"]
# unfreeze_list = ["inception_5b/5x5", "inception_5b/3x3", "inception_5b/1x1", "inception_5b/pool_proj",
#                  "inception_5b/3x3_reduce", "inception_5b_5x5_reduce"]