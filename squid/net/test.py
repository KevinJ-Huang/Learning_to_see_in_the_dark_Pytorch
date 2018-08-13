# import tensorflow as tf
# import torch
import numpy as np
import rawpy
import scipy.io


# norm = tf.random_normal([1,2,2,1],mean=0,stddev = 1)
# trans = tf.space_to_depth(norm,2)
#
# with tf.Session() as sess:
#     norm = sess.run(norm)
#     trans = sess.run(trans)
#
# print(norm)
# print(trans)

# def space_to_depth(input,block_size):
#     block_size_sq = block_size * block_size
#     (batch_size, s_height, s_width, s_depth) = input.size()
#     d_depth = s_depth * block_size_sq
#     d_width = int(s_width/block_size)
#     d_height = int(s_height/block_size)
#     t_1 = input.split(block_size,2)
#     stack = [t_t.contiguous().view(batch_size,d_height,d_depth) for t_t in t_1]
#     output = torch.stack(stack, 1)
#     output = output.permute(0,2,1,3)
#     return output
#
# norm = torch.rand(1,2,2,1)
# trans = space_to_depth(norm,2)
# print(trans.shape)


# a = np.array([[[1,2,3,4],[5,6,7,8]],[[10,11,12,13],[14,15,16,17]]])
# b = np.transpose(a,(2,1,0))
# print(a.shape)
# print(b.shape)
# print(b)



