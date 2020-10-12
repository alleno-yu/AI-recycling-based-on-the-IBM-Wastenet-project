from tensorflow.keras.layers import Layer

# developed by joelouismarino, and provided in this link: https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14

class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:,:,1:,1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
