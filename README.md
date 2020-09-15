# Generalized Neural Network (GNN)
This bundle provides a generalized ECL interface to Keras over Tensorflow.

It provides Keras / Tensorflow operations parallelized over an HPCC cluster.

Tensorflow Models are created transparently on each HPCC node, and training, evaluation and predictions
are done in a distributed fashion across the HPCC cluster.

GNN is designed to handle any type of Neural Network model that can be built
using Keras.  This includes Classical (Dense) Neural Networks as
well as Convolutional and Recursive Networks (such as LSTM), or any combination
of the above. 

GNN currently supports both Tensorflow 1.x and Tensorflow 2.x versions. It also supports the use of
GPUs in conjunction with Tensorflow, with certain
restrictions in the supported topology.  Specifically:
- All servers in a cluster must have the same GPU configuration
- The number of HPCC nodes must equal the number of GPUs.

One GPU will be allocated to each HPCC node. See GNNI module documentation for details.

The Module GNNI defines the ECL interface to Keras.  It supports any Keras
model (Functional or Sequential), and allows models with multiple inputs
and outputs.

Input to GNNI is in the form of Tensor records.  The built-in Tensor module defines
these Tensors and provides functions for operating on them.  Tensors provide an
efficient N-dimensional representation for data into and out of GNNI.

## INSTALLATION
Python3 and Tensorflow must be installed on each server running HPCC Systems Platform
software.  Tensorflow should be installed
using su so that all users can see it, and must be installed using the same version of
Python3 as is embedded in the HPCC Systems platform.
The file Test/SetupTest.ecl can be used to test the environment.  It will verify
that Python3 and Tensorflow are correctly installed on each Thor node.  To Install GNN, run:

> ecl bundle install https://github.com/hpcc-systems/GNN.git

## EXAMPLES
The files Test/ClassicTest.ecl and ClassificationTest.ecl show annotated examples
of using GNN to create a simple Classical Neural Networks using the Keras Sequential
model.

The file Test/FuncModelTest.ecl shows an example of building a classical regression /
classification network with multiple inputs and outputs using the Keras Functional
model.

The folder Test/HARTests
contains tests that show how to create more sophisticated Convolutional and
Recurrent networks.

## OTHER DOCUMENTATION
Programmer Documentation is available at:
[HPCC Machine Learning Library](http://hpccsystems.com/download/free-modules/machine-learning-library)
A tutorial on installing and running GNN is available at:
[Generalized Neural Network Blog](http://hpccsystems.com/blog/gnn-bundle)