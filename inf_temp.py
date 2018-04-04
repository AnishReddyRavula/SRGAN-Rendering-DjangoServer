import time
import time
start_time = time.time()
import tensorflow as tf
from lib.model import save_images
from lib.ops import *
import scipy.misc as sic
import numpy as np
import os
import math
import time
import collections
print("--- %s seconds ---" % (time.time() - start_time))







def inference_data_loader(image):
    # Get the image name list
    # if (INPUTDIR == 'None'):
    #     raise ValueError('Input directory is not provided')

    # if not os.path.exists(INPUTDIR):
    #     raise ValueError('Input directory not found')

    # image_list_LR_temp = os.listdir(INPUTDIR)
    # image_list_LR = [os.path.join(INPUTDIR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'png']

    # Read in and preprocess the images
    def preprocess_test(image):
        im = image
        # check grayscale image
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)
            temp[:, :, :] = im[:, :, np.newaxis]
            im = temp.copy()
        im = im / np.max(im)

        return im

    image_LR = [preprocess_test(image)]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_LR, inputs')

    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )






def generator(gen_inputs, gen_output_channels, reuse=False ):
    # Check the flag
    is_training = False
    res_blocks = 16

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, is_training)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, is_training)
            net = net + inputs

        return net


    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, res_blocks+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, is_training)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net





def save_images(fetches, step=None):
    OUTPUT_DIR = './result/'
    image_dir = os.path.join(OUTPUT_DIR, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = fetches['path_LR']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {"name": name, "step": step}

   
    kind = "outputs"
    filename = name + ".png"
    if step is not None:
        filename = "%08d-%s" % (step, filename)
    fileset[kind] = filename
    out_path = os.path.join(image_dir, filename)
    contents = fetches[kind][0]
    with open(out_path, "wb") as f:
        f.write(contents)
    filesets.append(fileset)
    return filesets





def index(image):
    CHECKPOINT = './SRGAN_pre-trained/model-200000'
    INPUTDIR = './MyImages/'
        # Check the checkpoint
    # if FLAGS.checkpoint is None:
    #     raise ValueError('The checkpoint file is needed to performing the test.')

    # In the testing time, no flip and crop is needed
    # if FLAGS.flip == True:
    #     FLAGS.flip = False

    # if FLAGS.crop_size is not None:
    #     FLAGS.crop_size = None

    # Declare the test data reader
    inference_data = inference_data_loader(image)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        # if FLAGS.task == 'SRGAN' or .task == 'SRResnet':
        gen_output = generator(inputs_raw, 3, reuse=False)
        

    print('Finish building the network')

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(inputs_raw)
        outputs = deprocess(gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope('encode_image'):
        save_fetch = {
            "path_LR": path_LR,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, CHECKPOINT)

        max_iter = len(inference_data.inputs)
        print('Evaluation starts!!')
        for i in range(max_iter):
            input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
            path_lr = inference_data.paths_LR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im.reshape(1, None, None, 3), path_LR: path_lr})
            filesets = save_images(results)
            print("--- %s seconds ---" % (time.time() - start_time))
            return filesets


    