# Fashion-MNIST-Pytorch
Convolutional network built using Pytorch to perform classification of Fashion MNIST dataset.

The Convolutional network architecture is as follows :

_```Conv2d Layer -> Relu -> MaxPool2d -> BatchNorm2d -> Conv2d Layer -> Relu -> MaxPool2d -> FullyConnected Layer Relu -> BatchNorm2d -> FullyConnected Layer Relu -> Output Layer```_

The Output Size is given by :

**If input is _n x n_**

_O = (n-f+2p/s)+1_

**If Input shape is non-square**

_Oh = nh-fh+2p/s +1_

_Ow = nw-fw+2p/s+1_
    
_`n -> input size`_

_`p -> padding`_

_`s -> stride length`_

_`f -> filter size`_