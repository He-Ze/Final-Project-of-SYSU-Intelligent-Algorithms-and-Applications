import numpy as np
import struct
import matplotlib.pyplot as plt

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton)
from PyQt5.QtGui import (QPainter, QPen)
from PyQt5.QtCore import Qt

import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

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

    def __init__(self, a_CNN):
        super(Example, self).__init__()

        self.resize(300, 300)
        self.setWindowTitle("Enter")

        self.setMouseTracking(False)

        self.pos_xy = []

        self.a_CNN = a_CNN

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

        self.setWindowTitle("%d" % torch.max(self.a_CNN(
            Variable(torch.reshape(torch.from_numpy(img_tmp / 255 * 0.99 + 0.01), [1, 1, 28, 28]).to(torch.float32))),
                                             1)[1][0])
        self.update()


class my_CNN(torch.nn.Module):
    def __init__(self):
        super(my_CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        return self.dense(res)


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

    my_net = my_CNN()
    optimizer = torch.optim.Adam(my_net.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    for _ in range(1):
        for i in range(len(train_images)):
            out = my_net(Variable(
                torch.reshape(torch.from_numpy(train_images[i] / 255 * 0.99 + 0.01), [1, 1, 28, 28]).to(torch.float32)))
            loss = loss_func(out, Variable(
                torch.reshape(torch.from_numpy(np.array([np.argmax(train_labels[i])])), [1]).long()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    my_net.eval()
    acc = []
    for i in range(len(test_images)):
        out = my_net(Variable(
            torch.reshape(torch.from_numpy(test_images[i] / 255 * 0.99 + 0.01), [1, 1, 28, 28]).to(torch.float32)))
        label = torch.max(out, 1)[1][0]

        if test_labels[i][label] == 0.99:
            acc.append(1)
        else:
            acc.append(0)

    print("1 acc is ", np.array(acc).mean())

    app = QApplication(sys.argv)
    pyqt_learn = Example(my_net)
    pyqt_learn.show()
    app.exec_()
