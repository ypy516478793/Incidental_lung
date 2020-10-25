from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from functools import reduce

import tensorflow as tf
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class DataGenerator():
    def __init__(self, data, batch_size):
        X, y = data
        assert (X.shape[0] == y.shape[0])
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]
        self.num_batches = X.shape[0] // self.batch_size
        if X.shape[0] % self.batch_size != 0:
            self.num_batches += 1
        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self, shuffle=True):
        if self.batch_index == self.num_batches:
            self.batch_index = 0
            if shuffle:
                indices = np.random.permutation(self.num_samples)
                self.X = self.X[indices]
                self.y = self.y[indices]
        start = self.batch_index * self.batch_size
        end = min(self.num_samples, start + self.batch_size)
        self.batch_index += 1
        return self.X[start: end], self.y[start: end]

class IncidentalData():
    def __init__(self, data_dir, image_size, num_classes):
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.load_data()
        self.split_data()

    def split_data(self):
        self.X = self.X[..., np.newaxis] / 255.0
        self.X = np.repeat(self.X, 3, axis=-1)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)

        print("Shape of train_x is: ", X_train.shape)
        print("Shape of train_y is: ", y_train.shape)
        print("Shape of test_x is: ", X_test.shape)
        print("Shape of test_y is: ", y_test.shape)

    def load_data(self, reload=False):
        data_path = os.path.join(self.data_dir, "2D_incidental_lung.npz")
        if os.path.exists(data_path) and not reload:
            self.data = np.load(data_path, allow_pickle = True)
            self.X, self.y = self.data["x"], self.data["y"]
        else:
            self.imageInfo = np.load(os.path.join(self.data_dir, "CTinfo.npz"), allow_pickle=True)["info"]
            pos_label_file = os.path.join(self.data_dir, "pos_labels.csv")
            self.pos_df = pd.read_csv(pos_label_file, dtype={"date": str})
            cat_label_file = os.path.join(self.data_dir, "Lung Nodule Clinical Data_Min Kim (No name).xlsx")
            self.cat_df = pd.read_excel(cat_label_file, dtype={"MRN": str})
            cat_key = [i for i in self.cat_df.columns if i.startswith("Category Of")][0]
            self.cats = self.cat_df[cat_key]
            self.screen()
            self.load_raw(data_path)

            exit(-1)

    def screen(self):
        num_images = len(self.imageInfo)
        mask = np.ones(num_images, dtype=bool)
        for imageId in range(num_images):
            pos = self.load_pos(imageId)
            cat = self.load_cat(imageId)
            if len(pos) > 1 and cat == 0:
                mask[imageId] = False
        self.imageInfo = self.imageInfo[mask]

    def load_raw(self, data_path):
        X, y = [], []
        for i in range(len(self.imageInfo)):
            slices = self.get_slices(i, self.image_size)
            label = self.load_cat(i)
            labels = np.eye(self.num_classes, dtype=np.int)[np.repeat(label, len(slices))]
            X.append(slices)
            y.append(labels)
        self.X = np.concatenate(X)
        self.y = np.concatenate(y)

        np.savez_compressed(data_path, x=self.X, y=self.y)
        print("Save slice 2D incidental lung nodule data to {:s}".format(data_path))

    def load_cat(self, imageId):
        imgInfo = self.imageInfo[imageId]
        patientID = imgInfo["patientID"]
        existId = (self.cat_df["MRN"].str.zfill(9) == patientID)
        cat = self.cats[existId].iloc[0]
        cat = int(cat > 2)
        return cat

    def load_image(self, imageId):
        from utils import lumTrans
        imgInfo = self.imageInfo[imageId]
        imgPath, thickness, spacing = imgInfo["imagePath"], imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        images = np.load(imgPath)["image"]
        images = lumTrans(images)
        print("Images{:d} shape: ".format(imageId), images.shape)
        return images

    def get_slices(self, imageId, size):
        from utils import extract_cube
        pos = self.load_pos(imageId)
        slices = []
        for i,p in enumerate(pos):
            images = self.load_image(imageId)
            cube = extract_cube(images, p, size=size)
            slices.append(cube[size//2])
        slices = np.array(slices)
        return slices

    def load_pos(self, imageId):
        from utils import resample_pos
        imgInfo = self.imageInfo[imageId]
        thickness, spacing = imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        pstr = imgInfo["pstr"]
        dstr = imgInfo["date"]
        existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
        pos = self.pos_df[existId][["x", "y", "z", "d"]].values
        pos = np.array([resample_pos(p, thickness, spacing) for p in pos])
        return pos


class Vgg19():
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.9, pretrain=False):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.not_load = ["fc8"] if pretrain else []

    def construct(self, istrain):
        """
        construct the model, set up the optimization
        :param istrain: boolean, True if train else False
        :return: None
        """
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.labels = tf.placeholder(tf.float32, [None, 2])
        self.train_mode = tf.placeholder(tf.bool)
        self.build(self.images, self.train_mode)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                               logits=self.logits)
        self.total_loss = tf.reduce_mean(self.loss)
        self.acc = tf.contrib.metrics.accuracy(labels=tf.argmax(self.labels, axis=-1),
                                               predictions=tf.argmax(self.prob, axis=-1))
        if istrain:
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.total_loss)

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 2, "fc8")
        self.logits = self.fc8
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict and name not in self.not_load:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

def train(train_loader, val_loader, model, sess, model_dir):
    istrain = True
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_auc_score = -np.inf
    best_acc = 0
    for e in range(args.epochs):
        n_batchs = train_loader.num_batches
        for b in range(n_batchs):
            x_batch, y_batch = next(train_loader)
            feed_dict = {model.images: x_batch,
                         model.labels: y_batch,
                         model.train_mode: istrain}
            output_tensor = [model.train_op, model.total_loss, model.acc]
            _, loss, acc = sess.run(output_tensor, feed_dict)
            print("epoch {:3d} | step {:3d} | training loss {:.2f} | training acc {:.2f}".format(e, b, loss, acc))
        train_losses.append(loss)
        train_accs.append(acc)

        # Run validation at the end of each epoch
        x_batch_val, y_batch_val = next(val_loader)
        feed_dict_val = {model.images: x_batch_val,
                         model.labels: y_batch_val,
                         model.train_mode: False}
        output_tensor = [model.total_loss, model.acc, model.labels, model.prob]
        loss_val, acc_val, labels_val, probs_val = sess.run(output_tensor, feed_dict_val)
        val_losses.append(loss_val)
        val_accs.append(acc_val)
        auc_score = roc_auc_score(labels_val[:, 1], probs_val[:, 1])

        print("epoch {:3d} | validation loss {:.2f} | validation acc {:.4f} | auc score {:.4f}".format(
              e, loss_val, acc_val, auc_score))

        if auc_score > best_auc_score or (auc_score == best_auc_score and val_accs > best_acc):
            save_path = os.path.join(model_dir, "vgg19_epoch{:d}.npy".format(e))
            model.save_npy(sess, save_path)
            best_auc_score = auc_score
            best_acc = val_accs

    print("Training process finished!")
    fig, ax = plt.subplots()
    ax.plot(np.array(train_losses), label="train loss")
    ax.plot(np.array(val_losses), label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("CE loss")
    plt.savefig(os.path.join(model_dir, "loss.png"), bbox_inches="tight", dpi=200)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(np.array(train_accs), label="train acc")
    ax.plot(np.array(val_accs), label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(model_dir, "acc.png"), bbox_inches="tight", dpi=200)
    plt.close()



def test(test_loader, model, sess, model_dir):
    x_batch_test, y_batch_test = next(test_loader)
    feed_dict_test = {model.images: x_batch_test,
                     model.labels: y_batch_test,
                     model.train_mode: False}
    output_tensor = [model.total_loss, model.acc, model.labels, model.prob]
    loss_test, acc_test, labels_test, probs_test = sess.run(output_tensor, feed_dict_test)
    auc_score = roc_auc_score(labels_test[:, 1], probs_test[:, 1])

    print("test loss {:.2f} | test acc {:.4f} | auc score {:.4f}".format(
        loss_test, acc_test, auc_score))

    all_label = np.argmax(labels_test, axis=-1)
    all_pred = np.argmax(probs_test, axis=-1)
    num_classes = args.num_classes
    confMat = confusion_matrix(all_label, all_pred)
    df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                         columns=[i for i in range(num_classes)])
    print("Test confusion matrix: ")
    print(df_cm)
    plt.figure()
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plt.savefig(os.path.join(model_dir, "test_confusion_matrix.png"), bbox_inches="tight", dpi=200)
    plt.close()

def main():
    dataset = IncidentalData(args.data_dir, args.image_size, args.num_classes)

    train_loader = DataGenerator(dataset.train_data, args.batchsize)
    test_loader = DataGenerator(dataset.test_data, batch_size=len(dataset.test_data[0]))

    imagenet_pretrain = False
    if args.load_model:
        load_path = args.load_model
    elif args.load_pretrain:
        load_path = "vgg19.npy"
        imagenet_pretrain = True
    else:
        load_path = None
    model = Vgg19(load_path, pretrain=imagenet_pretrain)
    model.construct(args.train)

    save_dir = args.save_dir
    model_name = 'bs_' + str(args.batchsize) + '.lr_' + str(args.lr) + args.extraStr
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if args.train:
        train(train_loader, test_loader, model, sess, model_dir)
    else:
        test(test_loader, model, sess, model_dir)


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datasource', type=str, help='SEAMII, BP2004', default="SEAMII")
    parser.add_argument('--data_dir', type=str, help='data file directory', default="../../data/")
    parser.add_argument('--save_dir', type=str, help="directory of saved results", default="results/")
    parser.add_argument('--epochs', type=int, help='number of epochs', default=30)
    parser.add_argument('--batchsize', type=int, help='batch size', default=16)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--image_size', type=int, help='image size', default=224)
    parser.add_argument('--num_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--train', type=eval, help='train or test', default=False)
    parser.add_argument('--load_pretrain', type=eval, help='whether to load pretrained model on imagenet', default=True)
    parser.add_argument('--gpu', type=str, help='which gpu to use', default="7")
    parser.add_argument('--load_model', type=str, help='trained model to load',
                        default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001_copy/vgg19_epoch22.npy")
    parser.add_argument('--extraStr', type=str, help='extraStr for saving', default="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main()