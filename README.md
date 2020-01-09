# Generalized Neural Network (GNN)
This bundle provides a generalized ECL interface to Keras over Tensorflow.

It provides Keras / Tensorflow operations parallelized over an HPCC cluster.

Models are created on each HPCC node, and training, evaluation and predictions
are done in a distributed fashion across the HPCC cluster.

The Module GNNI defines the ECL interface to Keras.  It currently only supports
the Keras Sequential model.

GNN is designed to handle any type of Neural Network model that can be built
using the Keras Sequential model.  This includes Classical Neural Networks as
well as Convolutional Networks and Recursive Networks such as LSTM, or any combination
of the above.

Input to GNNI is in the form of Tensor records.  The built-in Tensor module defines
these Tensors and provides functions for operating on them.  Tensors provide an
efficient N-dimensional representation for data into and out of GNNI.

## INSTALLATION
Python3 and Tensorflow must be installed on each server running HPCC Systems Platform
software.  Tensorflow should be installed
using su so that all users can see it, and must be installed using the same version of
Python3 as is embedded in the HPCC Systems platform.
The file Test/SetupTest.ecl can be used to test the environment.  It will verify
that Python3 and Tensorflow are correctly installed on each Thor node.

## EXAMPLES
The files Test/ClassicTest.ecl and ClassificationTest.ecl show annotated examples
of using GNN to create a simple Classical Neural Networks.  The folder Test/HARTests
contains tests that show how to create more sophisticated Convolutional and
Recurrent networks.

## OTHER DOCUMENTATION
Programmer Documentation is available at:
[HPCC Machine Learning Library](http://hpccsystems.com/download/free-modules/machine-learning-library)
A tutorial on installing and running GNN is available at:
[Generalize Neural Network Blog](http://hpccsystems.com/blog/gnn-bundle)