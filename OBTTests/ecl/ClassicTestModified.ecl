/*##############################################################################
    
    HPCC SYSTEMS software Copyright (C) 2022 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       
       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

// Modified version of the ClassicTest file that is compatible with the OBT

/**
  * Test GNNI with a classic regression neural network.
  * This test uses synthetic data generated internally.
  * It shows how GNN is used for regression using a simple
  * 2 dimensional (classic) input and output tensors.
  */

IMPORT Python3 AS Python;
IMPORT GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;

t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

RAND_MAX := POWER(2,32) -1;

// Test parameters
trainCount := 10000;
testCount := 1000;
featureCount := 5;
batchSize := 1024;
numEpochs := 10;
trainToLoss := .0001;
bsr := .25; // BatchSizeReduction.  1 = no reduction.  .25 = reduction to 25% of original.
lrr := 1.0;  // Learning Rate Reduction.  1 = no reduction.  .1 = reduction to 10 percent of original.
// END Test parameters

// Prepare training data.
// We use 5 inputs (X) and a single output (Y)
trainRec := RECORD
  UNSIGNED8 id;
  SET OF REAL x;
  REAL4 y;
END;

// The target function maps a set of X features into a Y value, which is a polynomial function of X.
REAL4 targetFunc(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt := .5 * POWER(x1, 4) - .4 * POWER(x2, 3) + .3 * POWER(x3,2) - .2 * x4 + .1 * x5;
  RETURN rslt;
END;

// Build the training data.  Pick random data for X values, and use a polynomial
// function of X to compute Y.
TrainSet := DATASET(trainCount, TRANSFORM(trainRec,
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
Train := PROJECT(TrainSet, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFunc(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Build the test data.  Same process as the training data.
TestSet := DATASET(testCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := 0)
                      );

Test := PROJECT(TestSet, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFunc(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Break the training and test data into X (independent) and Y (dependent) data sets.  Format as Tensor Data.
TrainInd := NORMALIZE(Train, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
TrainDep := NORMALIZE(Train, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));

// Form a Tensor from the tensor data.  This packs the data into 'slices' that can contain dense
// or sparse portions of the Tensor.  If the tensor is small, it will fit into a single slice.
// Huge tensors may require many slices.  The slices also contain tensor metadata such as the shape.
// For record oriented data, the first component of the shape should be 0, indicating that it is an
// arbitrary length set of records.
TrainIndTensor:= Tensor.R4.MakeTensor([0, featureCount], TrainInd);
TrainDepTensor := Tensor.R4.MakeTensor([0, 1], TrainDep);

TestInd := NORMALIZE(test, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
TestDep := NORMALIZE(test, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));

TestIndTensor := Tensor.R4.MakeTensor([0, featureCount], TestInd);
TestDepTensor:= Tensor.R4.MakeTensor([0, 1], TestDep);


// ldef provides the set of Keras layers that form the neural network.  These are
// provided as strings representing the Python layer definitions as would be provided
// to Keras.  Note that the symbol 'tf' is available for use (import tensorflow as tf),
// as is the symbol 'layers' (from tensorflow.keras import layers).
// Recall that in Keras, the input_shape must be present in the first layer.
// Note that this shape is the shape of a single observation.
ldef := ['''layers.Dense(256, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.Dense(1, activation=None)'''];

// compileDef defines the compile line to use for compiling the defined model.
// Note that 'model.' is implied, and should not be included in the compile line.
compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.02),
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
Session := GNNI.GetSession();

// DefineModel is dependent on the Session
//   ldef contains the Python definition for each Keras layer
//   compileDef contains the Keras compile statement.
Model := GNNI.DefineModel(Session, ldef, compileDef);

// Fit trains the models, given training X and Y data.  BatchSize is not the Keras batchSize,
// but defines how many records are processed on each node before synchronizing the weights
Model2 := GNNI.Fit(Model, TrainIndTensor, TrainDepTensor, batchSize := batchSize, numEpochs := numEpochs,
                      trainToLoss := trainToLoss, learningRateReduction := lrr,
                      batchSizeReduction := bsr);

// Predict computes the neural network output given a set of inputs.
Prediction := GNNI.Predict(Model2, TestIndTensor);

// Note that the Tensor is a packed structure of Tensor slices.  GetData()
// extracts the data into a sparse cell-based form -- each record represents
// one Tensor cell.  See Tensor.R4.TensData.
TestDepData := Tensor.R4.GetData(TestDepTensor);
PredData := Tensor.R4.GetData(Prediction);

// Here  we are comparing the expected values (testYDat) with the predicted values (predDat)
// and calculating the error and squared error to validate the process
cmp := JOIN(TestDepData, PredData, LEFT.indexes[1] = RIGHT.indexes[1], TRANSFORM({SET OF REAL indexes,
                                              REAL pred, REAL actual, REAL error, REAL sqError},
                  SELF.indexes := LEFT.indexes, SELF.pred := RIGHT.value, SELF.actual := LEFT.value,
                  SELF.error := ABS(SELF.actual - SELF.pred),
                  SELF.sqError := SELF.error * SELF.error), LEFT OUTER, LOCAL);

Result := RECORD
  REAL err := cmp.error;
END;

Errors := TABLE(cmp, Result);

// No errors should be above 0.1
HighErrors := COUNT(Errors(err > 0.1));

OUTPUT(IF(HighErrors = 0, 'Pass', 'Fail, ' + HighErrors + ' errors too high. Make sure your version of TensorFlow is >= 2.9.1.'), NAMED('Result'));
