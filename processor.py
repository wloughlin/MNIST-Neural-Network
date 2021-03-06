import random
import struct
import numpy as np

IMG_OFFSET = 16
LBL_OFFSET = 8


class processor(object):

    def __init__(self, imgs, labels):
        self.img_file = open(imgs)
        self.lbl_file = open(labels)
        self.preprocess()


        self.img_file.seek(4)
        self.lbl_file.seek(4)
        self.size = struct.unpack('>i', self.img_file.read(4))[0]
        self.rows = struct.unpack('>i', self.img_file.read(4))[0]
        self.cols = struct.unpack('>i', self.img_file.read(4))[0]
        self.img_size = self.rows * self.cols
        pixels = np.fromfile(self.img_file, dtype=np.uint8)
        pictures = [pixels[self.img_size * x: self.img_size *
                           (x + 1)] for x in range(self.size)]
        labels = np.fromfile(self.lbl_file, dtype=np.uint8)
        labels = labels[LBL_OFFSET - 4:]
        self.data = [(np.reshape(pictures[x], (self.img_size, 1)), labels[x])
                     for x in range(self.size)]

    def make_minibatches(self, num_batches):
        batches = [[] for i in range(num_batches)]
        for x in self.data:
            batches[random.randint(0, num_batches - 1)].append(x)
        return batches

    def shuffle_data(self):
        self.data = random.shuffle(self.data)

    def get_point(self):
        return self.data[random.randint(0, self.size)]
