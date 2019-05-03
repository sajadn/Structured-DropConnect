import math
import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.keras.layers.convolutional import Conv
import tensorflow as tf
import random


class BasicSpiralConv2D(Conv):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 training=True,
                 spiral_prob=None,
                 plus=False,
                 **kwargs):
        self.training = training
        self.spiral_prob = spiral_prob
        self.plus = plus

        super(BasicSpiralConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

    def create_spiral_constants(self, shape, spiral_numbers, training=True):

        '''
        :param  spiral
                shape: #output*input*height*width
        :return:
                train:
                    shape: spiral_number*input*height*width
                test:
                    shape: output*spiral_number*input*height*width
        '''

        height = shape[2]
        width = shape[3]
        spiral_kernels = []

        for s in range(spiral_numbers):

            if self.plus == True and training == True:
                const = np.ones(shape[2:], dtype=np.float32)
                for h in range(height):
                    for w in range(width):
                        if h < s or h > (height - s - 1):
                            const[h, :] = 0
                        elif w < s or w > (width - s - 1):
                            const[:, w] = 0
                spiral_kernels.append(const)
            elif self.plus == False or training == False:
                const = np.ones(shape[1:], dtype=np.float32)
                for h in range(height):
                    for w in range(width):
                        if h < s or h > (height - s - 1):
                            const[:, h, :] = 0
                        elif w < s or w > (width - s - 1):
                            const[:, :, w] = 0
                spiral_kernels.append(const)

        if training:
            return np.array(spiral_kernels)
        elif not training:
            return np.repeat(spiral_kernels[:-1], repeats=shape[0], axis=1).reshape(((spiral_numbers-1), *shape))



    # def create_spiral_coeff(self, spiral_numbers):
    #     if spiral_numbers == 2:
    #         return 2.
    #     elif spiral_numbers == 3:
    #         return 1.89

    def create_categorical_dist(self, spiral_numbers):
        maxprob = tf.constant(1.)
        remain = (maxprob - self.spiral_prob)*10
        if spiral_numbers == 2:
            return tf.stack([self.spiral_prob*10, remain])
        elif spiral_numbers == 3:
            return tf.stack([self.spiral_prob*10, remain/2, remain/2])

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.original_kernel = self.add_weight(
            name='original_kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
            self.input_spec = InputSpec(ndim=self.rank + 2,
                                        axes={channel_axis: input_dim})
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
            self._convolution_op = nn_ops.Convolution(input_shape,
                                                      filter_shape=self.original_kernel.get_shape(),
                                                      dilation_rate=self.dilation_rate,
                                                      strides=self.strides,
                                                      padding=op_padding.upper(),
                                                      data_format=conv_utils.convert_data_format(self.data_format,
                                                                                                 self.rank + 2))


        height = kernel_shape[1]
        self.spiral_numbers = int(math.ceil(height / 2.))+1
        spiral_constants = self.create_spiral_constants(kernel_shape[::-1], self.spiral_numbers, training=self.training)
        self.tf_spiral_constants = tf.constant(spiral_constants, name='spiral_constants')
        self.prob = self.create_categorical_dist(self.spiral_numbers)
        self.normalized_prob = (self.prob/tf.reduce_sum(self.prob))[:-1]
        tf.identity(self.normalized_prob, name='normalized_prob')
        self.normalized_prob = tf.reshape(tensor=self.normalized_prob, shape=[self.normalized_prob.shape[0], 1, 1, 1, 1])
        if self.plus == False:
            self.random_indices = tf.transpose(tf.multinomial(tf.log([self.prob]), self.filters), [1,0])
            print('random indices shape', self.random_indices.shape)
        elif self.plus == True:
            print("PPPPPPLUUUUUUSSSSSSS")
            self.random_indices = tf.multinomial(tf.log([self.prob]*self.filters), input_dim)
            self.random_indices = tf.expand_dims(self.random_indices, 2, 'random_index')
            print('random indices shape', self.random_indices.shape)

        self.build = True

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def call(self, inputs):
        self.my_kernel = tf.identity(self.original_kernel, name='my_kernel')

        if self.training == True:
            self.dropout_kernel = tf.gather_nd(self.tf_spiral_constants, self.random_indices, name='dropout_kernel')
            self.dropout_kernel = tf.transpose(self.dropout_kernel, [3, 2, 1, 0])
            self.kernel = tf.multiply(self.dropout_kernel, self.my_kernel, name='conv_kernel')
            return super(BasicSpiralConv2D, self).call(inputs)

        elif self.training == False:
            print("EVALUATING")
            self.dropout_kernel = tf.transpose(self.tf_spiral_constants, [0, 4, 3, 2, 1])*self.normalized_prob
            self.my_spiral_kernels = tf.multiply(self.my_kernel, self.dropout_kernel)
            #print('spiral shape', self.my_spiral_kernels.shape)
            self.my_spiral_kernels=tf.transpose(self.my_spiral_kernels, [0, 4, 3, 2, 1])
            msk_shape = self.my_spiral_kernels.shape
            final_output_channels = msk_shape[0]*msk_shape[1]
            self.kernel = tf.reshape(tensor=self.my_spiral_kernels, shape=(final_output_channels, *msk_shape[2:]))
            #print('kernel shape', self.kernel.shape)
            #print('input shape', inputs.shape)
            #print('stides', self.strides)
            self.kernel = tf.transpose(self.kernel, [3, 2, 1, 0])
            output = super(BasicSpiralConv2D, self).call(inputs)
            #print('output shape', output.shape)
            new_shape = (-1, (self.spiral_numbers-1), output.shape[1]//(self.spiral_numbers-1), *output.shape[-2:])
            #print('new shape', new_shape)
            output = tf.reshape(tensor=output, shape=new_shape)
            #print('new shaped', output.shape)

            layer_output = tf.reduce_sum(output, axis=1)
            #print(layer_output.shape)
            return layer_output


class SpiralConv2D(BasicSpiralConv2D, base.Layer):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 spiral_prob=None,
                 **kwargs):
        super(SpiralConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            spiral_prob=spiral_prob,
            **kwargs)


def spiral_conv2d(inputs,
                  filters,
                  kernel_size,
                  strides=(1, 1),
                  padding='valid',
                  data_format='channels_last',
                  dilation_rate=(1, 1),
                  activation=None,
                  use_bias=True,
                  kernel_initializer=None,
                  bias_initializer=init_ops.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  trainable=True,
                  name=None,
                  training=True,
                  spiral_prob=None,
                  reuse=None):

    layer = SpiralConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        training=training,
        spiral_prob=spiral_prob,
        _reuse=reuse,
        _scope=name,
    )
    return layer.apply(inputs)