import numpy as np

values = np.load('values.npy')
original_kernel = values[()]['resnet_model/spiral_conv2d_18/my_kernel']
dropout_kernel = values[()]['resnet_model/spiral_conv2d_18/dropout_kernel']
final_kernel = values[()]['resnet_model/spiral_conv2d_18/conv_kernel']
random_index = values[()]['resnet_model/spiral_conv2d_18/random_index']


original_kernel = np.abs(np.round(original_kernel, decimals=2))
final_kernel = np.abs(np.round(final_kernel, decimals=2))
dropout_kernel = np.abs(np.round(dropout_kernel, decimals=2))


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

#final_kernel = values[()]['resnet_model/spiral_conv2d_18/conv_kernel']

ind = 10
# for i in range(len(dropout_kernel)):
#     if dropout_kernel[i][0][0][0]==1:
#         print(i)
print(original_kernel.shape)
original_kernel = np.transpose(original_kernel, (3,2,1,0))[ind]

final_kernel = np.transpose(final_kernel, (3,2,1,0))[ind]
dropout_kernel = dropout_kernel[ind]

print(original_kernel.shape)
for i in range(len(original_kernel)):
    print('original')
    for j in range(len(original_kernel[i])):
        print(original_kernel[i][j])
    print(original_kernel[i].shape)
    print('dropout kernel')
    for j in range(len(dropout_kernel[i])):
        print(dropout_kernel[i][j])
    print('final kernel')
    for j in range(len(final_kernel[i])):
        print(final_kernel[i][j])
    print(final_kernel[i].shape)




# import numpy as np
# import tensorflow as tf
# from dropblock import DropBlock
# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#
# # only support `channels_last` data format
# a = tf.placeholder(tf.float32, [None, 10, 10, 3])
# keep_prob = tf.placeholder(tf.float32)
# training = tf.placeholder(tf.bool)
#
# drop_block = DropBlock(keep_prob=keep_prob, block_size=3)
# b = drop_block(a, training)
#
# sess = tf.Session()
# feed_dict = {a: np.ones([2, 10, 10, 3]), keep_prob: 0.5, training: True}
# c = sess.run(b, feed_dict=feed_dict)
#
# print(np.abs(np.around(c[0, :, :, 0], decimals=2)))