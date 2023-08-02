/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Test GNNI with a classic classification neural network.
  * This test uses synthetic data generated internally.
  * It shows how GNN is used for classification using a simple
  * 2 dimensional (classic) input and output.
  * GNNI supports NumericField matrices as input and output
  * for 2D problems, as well as the more flexible Tensor
  * input and output.
  * This test (arbitrarily) uses NumericField I/O instead of
  * Tensors in order to validate and illustrate this capability.
  * It could have used 2D Tensors.  Use of 2D Tensors is
  * demonstrated in ClassicTest.ecl.
  * For a classification example using higher dimensional tensors
  * and more complex neural networks, see:
  * HARTests/harLSTM.ecl or HARTests/harCNN_LSTM.ecl.
  * These examples also illustrate how to use the one-hot encoding
  * / decoding utilities.
  */
IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT ML_Core AS mlc;

NumericField := mlc.Types.NumericField;

// Prepare training data
RAND_MAX := POWER(2,32) -1;

// Test parameters
trainCount := 1000;
testCount := 100;
featureCount := 5;
classCount := 3;
numEpochs := 5;
batchSize := 128;
// End of Test Parameters

// Prepare training data.
// We use 5 inputs (X) and a one hot encoded output (Y) with 3 classes
// (i.e. 3 outputs).
trainRec := RECORD
  UNSIGNED8 id;
  SET OF REAL4 x;
  SET OF REAL4 y;
END;

// The target function maps a set of X features into a Y value,
// which is a threshold on a polynomial function of X.
// Note that we are effectively doing a One Hot encoding here, since we
// return a set of Y values, one for each class, with only one value
// being one and the rest zero.
// If we were working with tensors here, we could have used a class
// label and then called Utils.ToOneHot to encode it.
SET OF REAL4 targetFunc(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt0 := TANH(.5 * POWER(x1, 4) - .4 * POWER(x2, 3) + .3 * POWER(x3,2) - .2 * x4 + .1 * x5);
  rslt := MAP(rslt0 > -.25 => [1,0,0], rslt0 < .25 => [0,1,0], [0,0,1]);
  RETURN rslt;
END;

// Build the training data
train0 := DATASET(trainCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1],
                      SELF.y := [])
                      );
// Be sure to compute Y in a second step.  Otherewise, the RANDOM() will be executed twice and the Y will be based
// on different values than those assigned to X.  This is an ECL quirk that is not easy to fix.
train := PROJECT(train0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFunc(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));
OUTPUT(train, NAMED('trainData'));

// Build the test data.  Same process as the training data.
test0 := DATASET(testCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := [])
                      );

test := PROJECT(test0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFunc(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Break the training and test data into X (independent) and Y (dependent) data sets.
// Format as NumericField data.
trainX := NORMALIZE(train, featureCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.x[COUNTER]));
trainY := NORMALIZE(train, classCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.y[COUNTER]));

OUTPUT(trainX, NAMED('X1'));
OUTPUT(trainY, NAMED('y1'));

testX := NORMALIZE(test, featureCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.x[COUNTER]));
testY := NORMALIZE(test, classCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.y[COUNTER]));


// ldef provides the set of Keras layers that form the neural network.  These are
// provided as strings representing the Python layer definitions as would be provided
// to Keras.  Note that the symbol 'tf' is available for use (import tensorflow as tf), as is
// the symbol 'layers' (from tensorflow.keras import layers).
ldef := ['''layers.Dense(16, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(16, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

// compileDef defines the compile line to use for compiling the defined model.
// Note that 'model.' is implied, and should not be included in the compile line.
compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
              ''';

// Note that the order of the GNNI functions is maintained by passing tokens returned from one call
// into the next call that is dependent on it.
// For example, s is returned from GetSession().  It is used as the input to DefineModels(...) so
// that DefineModels() cannot execute until GetSession() has completed.
// Likewise, mod, the output from GetSession() is provided as input to Fit().  Fit in turn returns
// a token that is used by GetLoss(), EvaluateMod(), and Predict(), which are only dependent on Fit()
// having completed, and are not order dependent on one another.

// GetSession must be called before any other functions
s := GNNI.GetSession();
// Define model is dependent on the Session
//   ldef contains the Python definition for each Keras layer
//   compileDef contains the Keras compile statement.
mod := GNNI.DefineModel(s, ldef, compileDef);
// GetWeights returns the initialized weights that have been synchronized across all nodes.
wts := GNNI.GetWeights(mod);

OUTPUT(wts, NAMED('InitWeights'));

// Fit trains the models, given training X and Y data.  BatchSize is not the Keras batchSize,
// but defines how many records are processed on each node before synchronizing the weights
// Note that we use the NF form of Fit since we are using NumericField for I / o.
mod2 := GNNI.FitNF(mod, trainX, trainY, batchSize := batchSize, numEpochs := numEpochs);

OUTPUT(mod2, NAMED('mod2'));

// GetLoss returns the average loss for the final training epoch
losses := GNNI.GetLoss(mod2);

// EvaluateNF computes the loss, as well as any other metrics that were defined in the Keras
// compile line.  This is the NumericField form of EvaluateMod.
metrics := GNNI.EvaluateNF(mod2, testX, testY);

OUTPUT(metrics, NAMED('metrics'));

// PredictNF computes the neural network output given a set of inputs.
// This is the NumericField form of Predict. Note that these predictions are
// effectively the probabilities for each class (as output from softmax in the
// final NN layer).  If we had used Tensors rather than NumericField, we
// could convert to a class label by using Utils.FromOneHot, or
// Utils.Probabilities2Class.
preds := GNNI.PredictNF(mod2, testX);

OUTPUT(testY, ALL, NAMED('testDat'));
OUTPUT(preds, NAMED('predictions'));

OUTPUT(IF(metrics[2].value>0.95, 'Pass', 'Fail'), NAMED('Accuracy'));