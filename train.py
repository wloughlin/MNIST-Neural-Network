#!/usr/bin/env python

import processor as proc
import net as net
import sys






x = proc.processor("tims", "tlbl")
nn = net.net(x)
nn.load_state(sys.argv[1])
nn.train(sys.argv[2])
nn.save_state(sys.argv[1])

