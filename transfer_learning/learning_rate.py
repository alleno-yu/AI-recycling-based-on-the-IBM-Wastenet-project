import math

# learning rate schedule
def step_decay(epoch):

    # learning rate config
    initial_lrate = 0.00014928718
    drop = 0.96
    epochs_drop = 8

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    print("learning rate is: {:.2e}".format(lrate))
    return lrate