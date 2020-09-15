/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Multiple Model Test
  * Tests the use of multiple Keras models at the same time.
  * This is a simplified combination of ClassicTest.ecl (Regression)
  * and ClassificationTest.ecl (Classification).
  * This test uses synthetic data generated internally.
  * It shows how GNN can be used to produce and consume multiple
  * models in the same work-unit.
  */
IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;
IMPORT ML_Core AS mlc;

NumericField := mlc.Types.NumericField;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

RAND_MAX := POWER(2,32) -1;

// Test parameters
trainCount := 1000;
testCount := 1000;
featureCount := 5;
numEpochs := 10;
batchSize := 128;
// END Test parameters

// Prepare training data.
// We use 5 inputs (X) and a single output (Y)
trainRecR := RECORD
  UNSIGNED8 id;
  SET OF REAL x;
  REAL4 y;
END;

// The target function maps a set of X features into a Y value, which is a polynomial function of X.
REAL4 targetFuncR(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt := .5 * POWER(x1, 4) - .4 * POWER(x2, 3) + .3 * POWER(x3,2) - .2 * x4 + .1 * x5;
  RETURN rslt;
END;

// Build the training data.  Pick random data for X values, and use a polynomial
// function of X to compute Y.
trainR0 := DATASET(trainCount, TRANSFORM(trainRecR,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := 0)
                      );
// Be sure to compute Y in a second step.  Otherwise, the RANDOM() will be executed twice and the Y will be based
// on different values than those assigned to X.  This is an ECL quirk that is not easy to fix.
trainR := PROJECT(trainR0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncR(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));
OUTPUT(trainR, NAMED('trainDataR'));

// Build the test data.  Same process as the training data.
testR0 := DATASET(testCount, TRANSFORM(trainRecR,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := 0)
                      );

testR := PROJECT(testR0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncR(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Break the training and test data into X (independent) and Y (dependent) data sets.  Format as Tensor Data.
trainRX0 := NORMALIZE(trainR, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
trainRY0 := NORMALIZE(trainR, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));

// Form a Tensor from the tensor data.  This packs the data into 'slices' that can contain dense
// or sparse portions of the Tensor.  If the tensor is small, it will fit into a single slice.
// Huge tensors may require many slices.  The slices also contain tensor metadata such as the shape.
// For record oriented data, the first component of the shape should be 0, indicating that it is an
// arbitrary length set of records.
trainRX := Tensor.R4.MakeTensor([0, featureCount], trainRX0);
trainRY:= Tensor.R4.MakeTensor([0, 1], trainRY0);

OUTPUT(trainRX, NAMED('X1'));
OUTPUT(trainRY, NAMED('y1'));

testRX0 := NORMALIZE(testR, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
testRY0 := NORMALIZE(testR, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));
testRX := Tensor.R4.MakeTensor([0, featureCount], testRX0);
testRY:= Tensor.R4.MakeTensor([0, 1], testRY0);


// ldef provides the set of Keras layers that form the neural network.  These are
// provided as strings representing the Python layer definitions as would be provided
// to Keras.  Note that the symbol 'tf' is available for use (import tensorflow as tf),
// as is the symbol 'layers' (from tensorflow.keras import layers).
// Recall that in Keras, the input_shape must be present in the first layer.
// Note that this shape is the shape of a single observation.
ldefR := ['''layers.Dense(256, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.Dense(1, activation=None)'''];

// compileDef defines the compile line to use for compiling the defined model.
// Note that 'model.' is implied, and should not be included in the compile line.
compileDefR := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=tf.keras.losses.mean_squared_error,
              metrics=[tf.keras.metrics.mean_squared_error])
              ''';

// Note that the order of the GNNI functions is maintained by passing tokens returned from
// one call into the next call that is dependent on it.
// For example, s is returned from GetSession().  It is used as the input to
// DefineModels(...) so
// that DefineModels() cannot execute until GetSession() has completed.
// Likewise, mod, the output from GetSession() is provided as input to Fit().  Fit in turn
// returns a token that is used by GetLoss(), EvaluateMod(), and Predict(),
// which are only dependent on Fit() having completed, and are not order
// dependent on one another.

// GetSession must be called before any other functions
s := GNNI.GetSession();
// DefineModel is dependent on the Session
//   ldef contains the Python definition for each Keras layer
//   compileDef contains the Keras compile statement.
modR := GNNI.DefineModel(s, ldefR, compileDefR);
OUTPUT(modR, NAMED('modR'));
// GetWeights returns the initialized weights that have been synchronized across all nodes.
wtsR := GNNI.GetWeights(modR);

OUTPUT(wtsR, NAMED('InitWeightsR'));

// Fit trains the models, given training X and Y data.  BatchSize is not the Keras batchSize,
// but defines how many records are processed on each node before synchronizing the weights
modR2 := GNNI.Fit(modR, trainRX, trainRY, batchSize := batchSize, numEpochs := numEpochs);

OUTPUT(modR2, NAMED('modR2'));

// GetLoss returns the average loss for the final training epoch
lossesR := GNNI.GetLoss(modR2);

// EvaluateMod computes the loss, as well as any other metrics that were defined in the Keras
// compile line.
metricsR := GNNI.EvaluateMod(modR2, testRX, testRY);

OUTPUT(metricsR, NAMED('metricsR'));

// Predict computes the neural network output given a set of inputs.
predsR := GNNI.Predict(modR2, testRX);

// Note that the Tensor is a packed structure of Tensor slices.  GetData()
// extracts the data into a sparse cell-based form -- each record represents
// one Tensor cell.  See Tensor.R4.TensData.
testRYDat := Tensor.R4.GetData(testRY);
predDatR := Tensor.R4.GetData(predsR);

OUTPUT(SORT(testRYDat, indexes), ALL, NAMED('testDatR'));
OUTPUT(predsR, NAMED('predictionsR'));
OUTPUT(SORT(predDatR, indexes), ALL, NAMED('predDatR'));

//**********************************************************************************
// Classification Model
//**********************************************************************************
// Prepare training data.
// We use 5 inputs (X) and a one hot encoded output (Y) with 3 classes
// (i.e. 3 outputs).
trainRecC := RECORD
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
SET OF REAL4 targetFuncC(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt0 := TANH(.5 * POWER(x1, 4) - .4 * POWER(x2, 3) + .3 * POWER(x3,2) - .2 * x4 + .1 * x5);
  rslt := MAP(rslt0 > -.25 => [1,0,0], rslt0 < .25 => [0,1,0], [0,0,1]);
  RETURN rslt;
END;

// Build the training data
trainC0 := DATASET(trainCount, TRANSFORM(trainRecC,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := [])
                      );
// Be sure to compute Y in a second step.  Otherewise, the RANDOM() will be executed twice and the Y will be based
// on different values than those assigned to X.  This is an ECL quirk that is not easy to fix.
trainC := PROJECT(trainC0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncC(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));
OUTPUT(trainC, NAMED('trainData'));

// Build the test data.  Same process as the training data.
testC0 := DATASET(testCount, TRANSFORM(trainRecC,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := [])
                      );

testC := PROJECT(testC0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncC(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Break the training and test data into X (independent) and Y (dependent) data sets.
// Format as NumericField data.
trainCX := NORMALIZE(trainC, featureCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.x[COUNTER]));
trainCY := NORMALIZE(trainC, 3, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.y[COUNTER]));

OUTPUT(trainCX, NAMED('TrainCX'));
OUTPUT(trainCY, NAMED('TrainCY'));

testCX := NORMALIZE(testC, featureCount, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.x[COUNTER]));
testCY := NORMALIZE(testC, 3, TRANSFORM(NumericField,
                            SELF.wi := 1,
                            SELF.id := LEFT.id,
                            SELF.number := COUNTER,
                            SELF.value := LEFT.y[COUNTER]));


// ldef provides the set of Keras layers that form the neural network.  These are
// provided as strings representing the Python layer definitions as would be provided
// to Keras.  Note that the symbol 'tf' is available for use (import tensorflow as tf), as is
// the symbol 'layers' (from tensorflow.keras import layers).
ldefC := ['''layers.Dense(16, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(16, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

// compileDef defines the compile line to use for compiling the defined model.
// Note that 'model.' is implied, and should not be included in the compile line.
compileDefC := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=tf.keras.losses.mean_squared_error,
              metrics=[tf.keras.metrics.mean_squared_error])
              ''';

// Note that the order of the GNNI functions is maintained by passing tokens returned from one call
// into the next call that is dependent on it.
// For example, s is returned from GetSession().  It is used as the input to DefineModels(...) so
// that DefineModels() cannot execute until GetSession() has completed.
// Likewise, mod, the output from GetSession() is provided as input to Fit().  Fit in turn returns
// a token that is used by GetLoss(), EvaluateMod(), and Predict(), which are only dependent on Fit()
// having completed, and are not order dependent on one another.

// Use the same session ID as the Regression Network
// Define model is dependent on the Session
//   ldef contains the Python definition for each Keras layer
//   compileDef contains the Keras compile statement.
modC := GNNI.DefineModel(s, ldefC, compileDefC);
OUTPUT(modC, NAMED('ModC'));
// GetWeights returns the initialized weights that have been synchronized across all nodes.
wtsC := GNNI.GetWeights(modC);

OUTPUT(wtsC, NAMED('InitWeightsC'));

// Fit trains the models, given training X and Y data.  BatchSize is not the Keras batchSize,
// but defines how many records are processed on each node before synchronizing the weights
// Note that we use the NF form of Fit since we are using NumericField for I / o.
modC2 := GNNI.FitNF(modC, trainCX, trainCY, batchSize := batchSize, numEpochs := numEpochs);

OUTPUT(modC2, NAMED('modC2'));

// GetLoss returns the average loss for the final training epoch
lossesC := GNNI.GetLoss(modC2);

// EvaluateNF computes the loss, as well as any other metrics that were defined in the Keras
// compile line.  This is the NumericField form of EvaluateMod.
metricsC := GNNI.EvaluateNF(modC2, testCX, testCY);

OUTPUT(metricsC, NAMED('metricsC'));

// PredictNF computes the neural network output given a set of inputs.
// This is the NumericField form of Predict. Note that these predictions are
// effectively the probabilities for each class (as output from softmax in the
// final NN layer).  If we had used Tensors rather than NumericField, we
// could convert to a class label by using Utils.FromOneHot, or
// Utils.Probabilities2Class.
predsC := GNNI.PredictNF(modC2, testCX);

OUTPUT(testCY, ALL, NAMED('testDatC'));
OUTPUT(predsC, NAMED('predictionsC'));
