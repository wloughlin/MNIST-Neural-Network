import processor
import numpy as np


NUM_HL = 2
HL_SIZE = 16
FINAL_SIZE = 10



def sigmoid(x):
	return exp(x)/(1+exp(x))


def cost(result, target):
	target_array = np.zeroes((FINAL_SIZE, 1))


class net(object):

	def __init__(self, proc):
		self.data_set = proc
		layer_size = [proc.img_size]
		for i in range(NUM_HL):
			layer_size.append(HL_SIZE)
		layer_size.append(FINAL_SIZE)
		self.w_and_b = [(10*(np.random.rand(
			layer_size[x+1], layer_size[x])-0.5), 10*(np.random.rand(
			layer_size[x], 1)-0.5)) for x in range(NUM_HL+1)]

	def feed_foward(x):
		activation = x
		for (weight, bias) in self.w_and_b:
			activation = sigmoid(weight*activation + bias)
		return activation