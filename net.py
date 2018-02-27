import numpy as np
import pickle


NUM_HL = 2
HL_SIZE = 16
FINAL_SIZE = 10


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cost(result, target):
    cost_array = (target - result) ** 2
    d_cost_array = 2 * (target - result)
    cost = np.sum(cost_array)
    return (cost_array, d_cost_array, cost)


class net(object):

    def __init__(self, proc):
        self.data_set = proc
        self.layer_size = [proc.img_size]
        for i in range(NUM_HL):
            self.layer_size.append(HL_SIZE)
        self.layer_size.append(FINAL_SIZE)
        self.w_and_b = [(10 * (np.random.rand(
            self.layer_size[x + 1], self.layer_size[x]) - 0.5), 10 * (
                np.random.rand(self.layer_size[x + 1], 1) - 0.5)) for x in range(NUM_HL + 1)]

    def feed_foward(self, x):
        activations = [(x, x)]
        activation = x
        for (weight, bias) in self.w_and_b:
            result = weight.dot(activation) + bias
            activation = sigmoid(result)
            activations = [(activation, result)] + activations
        return activations

    def train(self, cycles):
        for cycle in range(int(cycles)):
            # self.data_set.shuffle_data()
            size = self.data_set.size
            del_weights = [np.zeros(
                (self.layer_size[x + 1], self.layer_size[x])) for x in range(NUM_HL + 1)]
            del_biases = [np.zeros((self.layer_size[x + 1], 1))
                          for x in range(NUM_HL + 1)]
            c = 0
            for (picture, label) in self.data_set.data:
                target = np.zeros((FINAL_SIZE, 1))
                activation = self.feed_foward(picture)
                target[label][0] += 1
                c += cost(activation[0][0], target)[2]
                changes = self.backprop(activation, target)
                del_weights = map(lambda y, z: y + z,
                                  map(lambda x: x[0], changes), del_weights)
                del_biases = map(lambda y, z: y + z,
                                 map(lambda x: x[1], changes), del_biases)

            del_weights = map(lambda x: x / size, del_weights)
            del_biases = map(lambda x: x / size, del_biases)
            for i in range(len(self.w_and_b)):
                w_b = self.w_and_b[i]
                w = w_b[0]
                b = w_b[1]
                w -= del_weights[i]
                b -= del_biases[i]
            if cycle % 10 == 0:
                print c / size

    def backprop(self, activations, target):
        ret = []
        d_cost = 2 * (activations[0][0] - target)
        for l in range(NUM_HL + 1):
            act_l = activations[l]
            act_l_minus = activations[l + 1]
            # calculate changes to weights and biases in current layer
            sp_z = d_sigmoid(act_l[1])
            change_b = sp_z * d_cost
            change_w = (change_b).dot(np.transpose(act_l_minus[0]))
            w_b = self.w_and_b[NUM_HL - l]
            # calculate cost for next layer
            w_trans = np.transpose(w_b[0])
            d_cost = w_trans.dot(change_b)
            ret = [(change_w, change_b)] + ret
        return ret

    def save_state(self, file_name):
        f = open(file_name, "w")
        pickle.dump(self.w_and_b, f)

    def load_state(self, file_name):
        f = open(file_name, "r")
        self.w_and_b = pickle.load(f)

    def guess(self):
        datum = self.data_set.get_point()
        results = self.feed_foward(datum[0])
        result = results[0][0]
        guess = (0, 0)
        for index in np.ndindex((10, 1)):
            if result[index] > result[guess]:
                guess = index
        if guess[0] == datum[1]:
            return 1
        return 0
