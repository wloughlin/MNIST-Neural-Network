#!/usr/bin/env python

import processor as proc
import net as net
import sys


x = proc.processor("tims", "tlbl")
nn = net.net(x)
nn.load_state(sys.argv[1])
correct = 0
for y in range(int(sys.argv[2])):
	correct += nn.guess()

print float(correct)/int(sys.argv[2])
