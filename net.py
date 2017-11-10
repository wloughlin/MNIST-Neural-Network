import processor
import numpy as np
import operator


NUM_HL = 2
HL_SIZE = 16
FINAL_SIZE = 10



def sigmoid(x):
	return 1/(1+np.exp(-x))

def d_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))


def cost(result, target):
	cost_array = (target - result)**2
	d_cost_array = 2*(target-result)
	cost = np.sum(cost_array)
	return (cost_array, d_cost_array, cost)


class net(object):

	def __init__(self, proc):
		self.data_set = proc
		layer_size = [proc.img_size]
		for i in range(NUM_HL):
			layer_size.append(HL_SIZE)
		layer_size.append(FINAL_SIZE)
		self.w_and_b = [(10*(np.random.rand(
			layer_size[x+1], layer_size[x])-0.5), 10*(np.random.rand(
			layer_size[x+1], 1)-0.5)) for x in range(NUM_HL+1)]

	def feed_foward(self, x):
		activations = []
		activation = x
		for (weight, bias) in self.w_and_b:
			result = weight.dot(activation) + bias
			activation = sigmoid(result)
			activations = [(activation, result)] + activations
		return activations

	def train(self):
		size = self.data_set.size/1000
		batches = self.data_set.make_minibatches(1000)
		for batch in batches:
			target = np.zeros((FINAL_SIZE, 1))
			activations = [np.zeros((FINAL_SIZE, 1)) for i in range(NUM_HL+1)]
			for (picture, label) in batch:
				test = self.feed_foward(picture)
				activations = map(operator.add, activations, test)
				target[label][0] += 1
			target = target/size
			activations = activations/size
			print cost(activations, target)[0]
			backprop(activations, target)




	def backprop(activations, target):

		d_cost = 2*(activations[0]-target)
		for l in range(NUM_HL+1):
			act_l = activations[l]
			act_l_minus = activations[l+1]
			# calculate changes to weights and biases in current layer
			sp_z = d_sigmoid(act_l[1])
			change_b = sp_z*d_cost
			change_w = (change_b).dot(act_l_minus[0])
			w_b = self.w_and_b[NUM_HL-l]
			#calculate cost for next layer
			w_trans = np.transpose(w_b[0])
			d_cost = w_trans.dot(change_b)
			w_b[0] += change_w
			w_b[1] += change_b




















