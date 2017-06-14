import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block

n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures used for CIFAR-10
def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    num_conv = (n - 20) / 12 + 1
    layers = []

    num_residual = 16

    chOut =0


    with tf.variable_scope('conv_start'):
        conv00 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv00)

    for lv in range(num_residual/2 -2 ):
        with tf.variable_scope('conv_%d' % (lv)):
            chOut += 168
            print layers[-1].get_shape().as_list()[1:]
            conv1_1_x = residual_block(layers[-1], 16+chOut,True)
            print conv1_1_x.get_shape().as_list()[1:]
            conv1_1 = residual_block(conv1_1_x,16+chOut,  False)

            layers.append(conv1_1_x)
            layers.append(conv1_1)


    #print layers[-1].get_shape().as_list()[1:]
    assert layers[-1].get_shape().as_list()[1:] == [7, 7, 1024]


    return layers[-1]
