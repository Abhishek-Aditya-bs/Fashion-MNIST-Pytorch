# Fashion-MNIST-Pytorch

`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend `Fashion-MNIST` to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example of how the data looks (*each class takes three-rows*):

![](https://github.com/Abhishek-Aditya-bs/Fashion-MNIST-Pytorch/blob/main/fashion-mnist-sprite.png)

<img src="https://github.com/Abhishek-Aditya-bs/Fashion-MNIST-Pytorch/blob/main/embedding.gif" width="100%">

## Get the Data

[Many ML libraries](#loading-data-with-other-machine-learning-libraries) already include Fashion-MNIST data/API, give it a try!

You can use direct links to download the dataset. The data is stored in the **same** format as the original [MNIST data](http://yann.lecun.com/exdb/mnist/).

| Name  | Content | Examples | Size | Link | MD5 Checksum|
| --- | --- |--- | --- |--- |--- |
| `train-images-idx3-ubyte.gz`  | training set images  | 60,000|26 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)|`8d4fb7e6c68d591d4c3dfef9ec88bf0d`|
| `train-labels-idx1-ubyte.gz`  | training set labels  |60,000|29 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)|`25c81989df183df01b3e8a0aad5dffbe`|
| `t10k-images-idx3-ubyte.gz`  | test set images  | 10,000|4.3 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)|`bef4ecab320f06d8554ea6380940ec79`|
| `t10k-labels-idx1-ubyte.gz`  | test set labels  | 10,000| 5.1 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)|`bb300cfdad3c16e7a12a480ee83cd310`|

## Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Load Dataset in Python 

To Load Dataset in Pytorch follow the documentation [here](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)

## Model Architecture

Convolutional network built using Pytorch to perform classification of Fashion MNIST dataset.

The Convolutional network architecture is as follows :

| Layer No. | Layer Name |
| --- | --- |
| 1 | _Conv2d Layer_ |
| 2 | _Relu_ |
| 3 | _MaxPool2d_ |
| 4 | _BatchNorm2d_ |
| 5 | _Conv2d Layer_ |
| 6 | _Relu_ |
| 7 | _MaxPool2d_ |
| 8 | _FullyConnected Layer_ |
| 9 | _BatchNorm2d_ |
| 9 | _FullyConnected Layer_ |
| 9 | _Relu_ |
| 9 | _Output Layer_ |

## The Output Size is given by :

**If input is _n x n_**

`_O = (n-f+2p/s)+1_`

**If Input shape is non-square**

`_Oh = nh-fh+2p/s +1_`

`_Ow = nw-fw+2p/s+1_`
    
where n -> input size,  
p -> padding,    
s -> stride length,   
f -> filter size.    
