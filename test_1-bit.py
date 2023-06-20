import torch
import torch.nn as nn

import os
import math

print('Verifiying 1-bit correctness ...')

B = 4
L = 512
M = 4096
N = 1024 #11008

DEV = torch.device('cuda:0')

from quant import *

torch.set_printoptions(precision=3)

layer = nn.Linear(M, N)

weight_shape = layer.weight.data.shape
bias_shape = layer.bias.data.shape

# layer.weight.data = torch.randint(-1, 2, weight_shape).float()
nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
layer.bias.data = torch.randint(0, 2, bias_shape).float()
#layer.weight.data = torch.ones( weight_shape).float()
#layer.bias.data = torch.ones(bias_shape).float()


#vec = torch.randint(0, 2, (B,L,M)).to(DEV).half()
vec = torch.randn(B,L,M).to(DEV).half()

layer.to(DEV)
print('pytorch:', layer(vec.float()))

print("float weights:")
print(layer.weight.data)

quantizer = Quantizer()
quantizer.configure(1, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
quantizer.scale.fill_(1)
quantizer.zero.fill_(0)

layer.weight.data = quantize(layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq)

print("quantized weights:")
print(layer.weight.data)

use_bias = layer.bias is not None
qlayer = QuantLinear(1, -1, layer.in_features, layer.out_features, use_bias)
#print('Scale / Zero', quantizer.scale, quantizer.zero)
# layer
qlayer.pack(layer, quantizer.scale, quantizer.zero)
# have to reset qlayer zero and scales

qlayer = qlayer.to(DEV)
layer = layer.to(DEV).half()

with torch.no_grad():
    gt = layer(vec)
    quantized_result = qlayer(vec)
    # print('1bit Simu:', gt, 'Min - Max', gt.min(), gt.max())
    # print('1bit Kern:', quantized_result, 'Min - Max', quantized_result.min(), quantized_result.max())
    # print(gt.shape)
    # print(quantized_result.shape)
    if torch.all(torch.isclose(gt, quantized_result, atol=1)):
        print("Inference result: equal")
    else:
        print("Inference result: not equal")