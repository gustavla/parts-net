__author__ = 'amit'
import pnet
import deepdish as dd
import numpy as np
import pylab as py
import net_test
import test_edge
import net_test


from pnet.oriented_gaussian_parts_layer import lognormalized


net=pnet.Layer.load('Enets/net76.h5')
#print('Mean error rate',np.mean(rates),'Std',np.std(rates));
for l in (1,1):
   visparts = net.layers[l].visualized_parts
   grid = dd.plot.ImageGrid(visparts.squeeze())
   grid.save('parts'+str(l)+'.png', scale=5)
#M=10
#net_test.net_test(net,M,numtest=1000)
#

# Settings=OrderedDict()
# net=pnet.Layer.load('Enets/net63.h5')
# l=np.int(0)
# for l, layer in enumerate(net._layers):
#     if (hasattr(layer,"_settings")):
#         Settings.update({layer.name: layer._settings})
#
# print(Settings.items())

