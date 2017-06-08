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



    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)
        print("conv1 shape:")
        print(conv1.get_shape())

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

	    print("conv2 shape:")
        print(conv2.get_shape())
        assert conv2.get_shape().as_list()[1:] == [448, 448, 16]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [224, 224, 32]
    
    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x = residual_block(layers[-1], 64, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [112, 112, 64]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv5_%d' % (i+1)):
            conv5_x = residual_block(layers[-1], 128, down_sample)
            conv5 = residual_block(conv5_x, 128, False)
            layers.append(conv5_x)
            layers.append(conv5)

        assert conv5.get_shape().as_list()[1:] == [56, 56, 128]

    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv6_%d' % (i+1)):
            conv6_x = residual_block(layers[-1], 256, down_sample)
            conv6 = residual_block(conv6_x, 256, False)
            layers.append(conv6_x)
            layers.append(conv6)

        assert conv6.get_shape().as_list()[1:] == [28, 28, 256]




    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv7_%d' % (i+1)):
            conv7_x = residual_block(layers[-1], 512, down_sample)
            conv7 = residual_block(conv7_x, 512, False)
            layers.append(conv7_x)
            layers.append(conv7)

        assert conv7.get_shape().as_list()[1:] == [14, 14, 512]


    for i in range (num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv8_%d' % (i+1)):
            conv8_x = residual_block(layers[-1], 1024, down_sample)
            conv8 = residual_block(conv8_x, 1024, False)
            layers.append(conv8_x)
            layers.append(conv8)

        assert conv8.get_shape().as_list()[1:] == [7, 7, 1024]

    return layers[-1]
