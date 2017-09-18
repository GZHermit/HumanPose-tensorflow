# coding:utf-8
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def layers(op):
    def decorated_layer(self, *args, **kwargs):

        layer_name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # print(layer_name)
        terminals_len = len(self.terminals)
        if terminals_len == 0:
            raise RuntimeError('No input variables found for layer %s.' % layer_name)
        elif terminals_len == 1:
            layer_inputs = self.terminals[0]
        else:
            layer_inputs = list(self.terminals)
        layer_output = op(self, layer_inputs, *args, **kwargs)
        self.layers[layer_name] = layer_output
        self.feed(layer_output)
        return self

    return decorated_layer


class NetWork(object):
    def __init__(self, input, nFeat=512, nStack=4, nModules=1, nLow=4,
                 outputDim=16, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000,
                 is_training=True):
        self.nStack = nStack
        self.nFeat = nFeat
        self.nModules = nModules
        self.outDim = outputDim
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.decay_step = decay_step
        self.nLow = nLow
        self.joints = ['r_anckle', 'r_knee', 'r_hip',
                       'l_hip', 'l_knee', 'l_anckle',
                       'pelvis', 'thorax', 'neck', 'head',
                       'r_wrist', 'r_elbow', 'r_shoulder',
                       'l_shoulder', 'l_elbow', 'l_wrist']
        self.input = input
        self.layers = dict(input)  # This dict includes all the layers in the model
        self.terminals = []  # This is the temporary list storing the layers
        self.setup(is_training)

    def setup(self, *args):
        pass

    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.terminals.append(layer)
        return self

    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def get_output(self):
        return self.terminals[-1]

    def make_var(self, name, shape):
        return tf.get_variable(name, shape, dtype=tf.float32)

    @layers
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layers
    def batch_normalization(self, input, is_training, name, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:
            import tensorflow.contrib.slim as slim
            output = slim.batch_norm(
                input,
                0.9,
                epsilon=1e-5,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    @layers
    def conv(self, input, kernel, out_channel, strides, name, padding='SAME', biased=True):
        in_channel = input.get_shape().as_list()[-1]

        with tf.variable_scope(name):
            kernel = self.make_var(name='weight', shape=[kernel[0], kernel[1], in_channel, out_channel])
            out = tf.nn.conv2d(input, kernel, strides=[1, strides[0], strides[1], 1], padding=padding)
            if biased:
                biases = self.make_var(name='bias', shape=[out_channel])
                out = tf.nn.bias_add(out, biases)
            return out

    @layers
    def dropout(self, input, rate, is_training, name):
        return tf.layers.dropout(input, rate, training=is_training, name=name)

    @layers
    def max_pool(self, input, kernel, strides, name, padding='SAME'):
        return tf.nn.max_pool(input,
                              ksize=[1, kernel[0], kernel[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding=padding,
                              name=name)

    @layers
    def pad(self, input, t2b, l2r, name):
        padding = np.array([[0, 0], t2b, l2r, [0, 0]])
        return tf.pad(input, padding, name=name)

    @layers
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layers
    def resize_nearest_neighbor(self, input, new_size, name):
        return tf.image.resize_nearest_neighbor(input, new_size, name=name)

    @layers
    def stack(self, inputs, axis, name):
        return tf.stack(inputs, axis, name)

    @layers
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name)
