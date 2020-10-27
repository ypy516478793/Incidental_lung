from keras.applications.vgg19 import (
    VGG19, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_image_array(img):
    '''

    :param image: shape == (224, 224)
    :return:
    '''
    x = image.img_to_array(img)
    x = np.repeat(x, 3, axis=-1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG19(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam_tf(sess, model, image, category_index, layer_name):

    nb_classes = 2
    model.target_logit = tf.multiply(model.logits, K.one_hot([category_index], nb_classes))
    loss = K.sum(model.target_logit)

    if layer_name == 'block5_conv3':
        conv_output = model.conv5_3
    elif layer_name == 'block5_conv4':
        conv_output = model.conv5_4

    grads = normalize(K.gradients(loss, conv_output)[0])

    feed_dict = {model.images: image,
                 model.train_mode: False}
    output_tensor = [conv_output, grads]

    output, grads_val = sess.run(output_tensor, feed_dict)
    output, grads_val = output[0, :], grads_val[0, :, :, :]


    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)

    return np.uint8(cam), heatmap


def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

if __name__ == '__main__':

    data_dir = "../../data/"
    save_dir = "gradcam"
    os.makedirs(save_dir, exist_ok=True)

    images_path = os.path.join(data_dir, "2D_incidental_lung.npz")
    images = np.load(images_path, allow_pickle = True)["x"]

    # preprocessed_input = load_image(sys.argv[1])
    from privacy.VGG2D.train_vgg import Vgg19

    load_path = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001_copy/vgg19_epoch22.npy"
    model = Vgg19(load_path, pretrain=False)
    model.construct(False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(len(images)):
        preprocessed_input = load_image_array(images[i])
        feed_dict_test = {model.images: preprocessed_input,
                         # model.labels: y_batch_test,
                         model.train_mode: False}
        # output_tensor = [model.total_loss, model.acc, model.labels, model.prob]
        output_tensor = model.prob
        predictions = sess.run(output_tensor, feed_dict_test)
        predicted_class = np.argmax(predictions)
        print('Sample {:d}, predicted class:'.format(i))
        print('{:d} with probability {:.2f}'.format(predicted_class, predictions.max()))

        # model = VGG19(weights='imagenet')
        # predictions = model.predict(preprocessed_input)
        # top_1 = decode_predictions(predictions)[0][0]
        # print('Predicted class:')
        # print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
        # predicted_class = np.argmax(predictions)

        # cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
        cam, heatmap = grad_cam_tf(sess, model, preprocessed_input, predicted_class, "block5_conv3")
        cv2.imwrite(os.path.join(save_dir, "gradcam_{:d}.jpg".format(i)), cam)
        cv2.imwrite(os.path.join(save_dir, "heatmap_{:d}.jpg".format(i)), (heatmap * 255).astype(np.int))

        # register_gradient()
        # guided_model = modify_backprop(model, 'GuidedBackProp')
        # saliency_fn = compile_saliency_function(guided_model)
        # saliency = saliency_fn([preprocessed_input, 0])
        # gradcam = saliency[0] * heatmap[..., np.newaxis]
        # cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))