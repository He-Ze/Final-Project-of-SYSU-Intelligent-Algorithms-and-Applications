import numpy as np
import struct
import matplotlib.pyplot as plt

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton)
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtCore import Qt

train_images_idx3_ubyte_file = 'tc/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = 'tc/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = 'tc/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'tc/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(
        image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.zeros((num_images, 10)) + 0.01
    for i in range(num_images):
        labels[i][struct.unpack_from(fmt_image, bin_data, offset)[0]] = 0.99

        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    print("Read train images...")
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    print("Read train labels...")
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    print("Read test images...")
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    print("Read test labels...")
    return decode_idx1_ubyte(idx_ubyte_file)


class Example(QWidget):

    def __init__(self, a_BP):
        super(Example, self).__init__()

        self.resize(300, 300)
        self.setWindowTitle("Enter")

        self.setMouseTracking(False)

        self.pos_xy = []

        self.a_BP = a_BP

        self.button = QPushButton("reset", self)
        self.button.clicked.connect(self.self_reset)

    def self_reset(self):
        self.setWindowTitle("Enter")
        self.pos_xy = []
        self.resize(300, 300)

        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        img_tmp = np.zeros((28, 28))
        for pos_tmp in self.pos_xy:
            if pos_tmp != (-1, -1):
                tx = int(pos_tmp[1] * 28 / self.size().height())
                ty = int(pos_tmp[0] * 28 / self.size().width())
                img_tmp[tx][ty] = 255

        # plt.imshow(img_tmp, cmap='gray')
        # plt.show()

        self.setWindowTitle("%d" % np.argmax(self.a_BP.predict(img_tmp.reshape(784) / 255 * 0.99 + 0.01)))
        self.update()


def active_f(x):
    return 1 / (1 + np.exp(-x))
    # return np.maximum(0, x)


def active_f_g(x):
    return active_f(x) * (1 - active_f(x))
    # return (x >= 0).astype(int)


class BP:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.__weight1 = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
        self.__weight2 = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))
        self.__learning_rate = learning_rate

    def train(self, t_input, t_target):
        inputs = np.array(t_input, ndmin=2).T
        targets = np.array(t_target, ndmin=2).T
        hidden_inputs = np.dot(self.__weight1, inputs)
        hidden_outputs = active_f(hidden_inputs)
        final_inputs = np.dot(self.__weight2, hidden_outputs)
        final_outputs = active_f(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.__weight2.T, output_errors)
        self.__weight2 += self.__learning_rate * np.dot((output_errors * active_f_g(final_inputs)),
                                                        np.transpose(hidden_outputs))
        self.__weight1 += self.__learning_rate * np.dot((hidden_errors * active_f_g(hidden_inputs)),
                                                        (np.transpose(inputs)))

    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.__weight1, inputs)
        hidden_outputs = active_f(hidden_inputs)
        final_inputs = np.dot(self.__weight2, hidden_outputs)
        final_outputs = active_f(final_inputs)
        return final_outputs


if __name__ == '__main__':

    train_images = load_train_images()

    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    # for i in range(100):
    #     plt.title(list(train_labels[i]).index(0.99))
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.pause(0.5)
    # plt.show()

    # num_hiddens = [10, 50, 100, 200, 300]
    #
    # accs = []
    #
    # for num_hidden in num_hiddens:
    #     my_net = BP(784, num_hidden, 10, 0.1)
    #
    #     for _ in range(10):
    #         for i in range(len(train_images)):
    #             my_net.train(train_images[i].reshape(784) / 255 * 0.99 + 0.01, train_labels[i])
    #
    #     acc = []
    #     for i in range(len(test_images)):
    #         label = np.argmax(my_net.predict(test_images[i].reshape(784) / 255 * 0.99 + 0.01))
    #         if test_labels[i][label] == 0.99:
    #             acc.append(1)
    #         else:
    #             acc.append(0)
    #
    #     accs.append(np.array(acc).mean())
    #     print("acc is ", np.array(acc).mean(), "when hidden num is ", num_hidden)
    #
    # plt.plot(num_hiddens, accs, color='b')
    # plt.xticks(num_hiddens)
    # plt.xlabel('Num of hidden nodes')
    # plt.ylabel('accuracy')
    # plt.title('sigmoid')
    # plt.legend()
    # plt.show()

    my_net = BP(784, 300, 10, 0.1)

    for i in range(len(train_images)):
        my_net.train(train_images[i].reshape(784) / 255 * 0.99 + 0.01, train_labels[i])

    acc = []
    for i in range(len(test_images)):
        label = np.argmax(my_net.predict(test_images[i].reshape(784) / 255 * 0.99 + 0.01))
        if test_labels[i][label] == 0.99:
            acc.append(1)
        else:
            acc.append(0)

    print("acc is ", np.array(acc).mean())

    app = QApplication(sys.argv)
    pyqt_learn = Example(my_net)
    pyqt_learn.show()
    app.exec_()

