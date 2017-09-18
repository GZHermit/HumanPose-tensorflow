# coding:utf-8
import tensorflow as tf
from models.network import NetWork


class Single(NetWork):
    def residual(self, output_channel, name, is_training=True):
        name += '/'
        input = self.get_output()
        (self.feed(input)
         .batch_normalization(name=name + 'norm_1', is_training=is_training, activation_fn=tf.nn.relu)
         .conv([1, 1], int(output_channel / 2), [1, 1], padding='VALID', name=name + 'conv_1')
         .batch_normalization(name=name + 'norm_2', is_training=is_training, activation_fn=tf.nn.relu)
         .pad([1, 1], [1, 1], name=name + 'pad')
         .conv([3, 3], int(output_channel / 2), [1, 1], padding='VALID', name=name + 'conv_2')
         .batch_normalization(name=name + 'norm_3', is_training=is_training, activation_fn=tf.nn.relu)
         .conv([1, 1], int(output_channel), [1, 1], padding='VALID', name=name + 'conv_3'))

        convblock = self.get_output()

        if input.get_shape().as_list()[-1] != output_channel:
            (self.feed(input)
             .conv([1, 1], int(output_channel), [1, 1], padding='VALID', name=name + 'conv_skip'))
            skiplayer = self.get_output()
        else:
            skiplayer = input

        (self.feed(convblock,
                   skiplayer)
         .add(name=name + 'add')
         .relu(name=name + 'relu'))
        return self

    def hourglass(self, output_channel, nLow, name):
        name += '/'
        self.residual(output_channel, name=name + 'res_up_1')
        up_1 = self.get_output()
        self.feed(up_1).max_pool([2, 2], [2, 2], name=name + 'max_pool')
        self.residual(output_channel, name=name + 'res_low_1')
        if nLow > 0:
            self.hourglass(nLow - 1, output_channel, name=name + 'hg_low_2')
        else:
            self.residual(output_channel, name=name + 'res_low_2')
        self.residual(output_channel, name=name + 'res_low_3')
        new_size = tf.shape(self.get_output())[1:3] * 2
        (self.feed(self.get_output())
         .resize_nearest_neighbor(new_size, name=name + 'up_2'))
        (self.feed(self.get_output(), up_1)
         .add(name=name + 'add')
         .relu(name=name + 'relu'))
        return self

    def setup(self, is_training):
        # Input Dim : nbImages x 256 x 256 x 3
        (self.feed('data')
         .pad([2, 2], [2, 2], name='pad_1')  # Dim pad1 : nbImages x 260 x 260 x 3
         .conv([6, 6], 64, [2, 2], padding='VALID', name='conv_1')  # Dim conv1 : nbImages x 128 x 128 x 64
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu))
        self.residual(128, 'r1', is_training)
        # Dim pad1 : nbImages x 128 x 128 x 128
        self.feed(self.get_output()).max_pool([2, 2], [2, 2], name='max_pool')
        # Dim pool1 : nbImages x 64 x 64 x 128
        self.residual(int(self.nFeat / 2), 'r2', is_training)
        self.residual(self.nFeat, 'r3')
        outs = []
        for index in range(self.nStack):
            primary_input = self.get_output()
            name = 'stage_%d/' % index

            self.hourglass(self.nFeat, self.nLow, name=name + 'hg')
            (self.feed(self.get_output())
             .dropout(rate=self.dropout_rate, is_training=is_training, name=name + 'dropout')
             .conv([1, 1], self.nFeat, [1, 1], padding='VALID', name=name + 'conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu))
            ll = self.get_output()
            (self.feed(ll)
             .conv([1, 1], self.outDim, [1, 1], padding='VALID', name=name + 'out')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu))
            outs.append(self.get_output())
            if index == self.nStack - 1: break
            (self.feed(self.get_output())
             .conv([1, 1], self.nFeat, [1, 1], padding='VALID', name=name + 'out_'))
            out_ = self.get_output()
            (self.feed(ll)
             .conv([1, 1], self.nFeat, [1, 1], padding='VALID', name=name + 'll_'))
            (self.feed(self.get_output(),
                       primary_input,
                       out_)
             .add(name=name + 'add'))
        print("hehe")
        (self.feed(*outs)
         .stack(axis=1, name='stack_output')
         .sigmoid(name='final_output'))
