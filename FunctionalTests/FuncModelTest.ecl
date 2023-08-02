/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Test GNNI Functional (i.e. non-sequential) Model Capability
  * Create a single model that does a Regression and a Classification in a single
  * functional model.  This model has two sets of inputs (one for Regression and
  * one for Classification), and two corresponding outputs.
  */
IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;
IMPORT GNN.Utils;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
FuncLayerDef := Types.FuncLayerDef;

RAND_MAX := POWER(2,32) -1;

// Test parameters
numEpochs := 10;
batchSize := 128;
trainCount := 1000;
testCount := 1000;
featureCount := 5;
// END Test parameters

// Prepare Regression training and test data.
// We use 5 inputs (X) and a single output (Y)
trainRecR := RECORD
  UNSIGNED8 id;
  SET OF REAL x;
  REAL4 y;
END;

// The Regression target function maps a set of X features into a Y value, which is a polynomial function of X.
REAL4 targetFuncR(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt := .5 * POWER(x1, 4) - .4 * POWER(x2, 3) + .3 * POWER(x3,2) - .2 * x4 + .1 * x5;
  RETURN rslt;
END;

// Build the regression training data.  Pick random data for X values, and use a polynomial
// function of X to compute Y.
train0R := DATASET(trainCount, TRANSFORM(trainRecR,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1],
                      SELF.y := 0)
                      );
// Be sure to compute Y in a second step.  Otherwise, the RANDOM() will be executed twice and the Y will be based
// on different values than those assigned to X.  This is an ECL quirk that is not easy to fix.
trainR := PROJECT(train0R, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncR(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));
OUTPUT(trainR, NAMED('trainDataR'));

// Build the test data.  Same process as the training data.
test0R := DATASET(testCount, TRANSFORM(trainRecR,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1],
                      SELF.y := 0)
                      );

testR := PROJECT(test0R, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncR(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Break the training and test data into X (independent) and Y (dependent) data sets.  Format as Tensor Data.
trainX0R := NORMALIZE(trainR, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
trainY0R := NORMALIZE(trainR, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));

// Form a Tensor from the tensor data.  This packs the data into 'slices' that can contain dense
// or sparse portions of the Tensor.  If the tensor is small, it will fit into a single slice.
// Huge tensors may require many slices.  The slices also contain tensor metadata such as the shape.
// For record oriented data, the first component of the shape should be 0, indicating that it is an
// arbitrary length set of records.
trainXR := Tensor.R4.MakeTensor([0, featureCount], trainX0R);
trainYR:= Tensor.R4.MakeTensor([0, 1], trainY0R);

OUTPUT(trainXR, NAMED('trainXR'));
OUTPUT(trainYR, NAMED('trainYR'));

testX0R := NORMALIZE(testR, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
testY0R := NORMALIZE(testR, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));
testXR := Tensor.R4.MakeTensor([0, featureCount], testX0R);
testYR:= Tensor.R4.MakeTensor([0, 1], testY0R);

// Now Prepare Classification training and test data.
// We use 5 inputs (X) and a one hot encoded output (Y) with 3 classes
// (i.e. 3 outputs).
trainRecC := RECORD
  UNSIGNED8 id;
  SET OF REAL4 x;
  REAL4 y;
END;

// The target function maps a set of X features into a Y value,
// which is a threshold on a polynomial function of X.

// Returns 1 of 3 classes, 0, 1, or 2.
REAL4 targetFuncC(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt0 := TANH(POWER(x1, 4) - 2 * POWER(x2, 3) + 3 * POWER(x3,2) - 4 * x4 + 5 * x5);
  rslt := MAP(rslt0 < -.33 => 0, rslt0 < .33 => 1, 2);
  RETURN rslt;
END;

// Build the training data
train0C := DATASET(trainCount, TRANSFORM(trainRecC,
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
trainC := PROJECT(train0C, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncC(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));
OUTPUT(trainC, NAMED('trainDataC'));

// Build the test data.  Same process as the training data.
test0C := DATASET(testCount, TRANSFORM(trainRecC,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1,
                                  (RANDOM() % RAND_MAX) / (RAND_MAX / 2) - 1],
                      SELF.y := [])
                      );

testC := PROJECT(test0C, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFuncC(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));

// Break the training and test data into X (independent) and Y (dependent) data sets.
// Format as NumericField data.
trainX0C := NORMALIZE(trainC, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
trainY0C := NORMALIZE(trainC, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.Y));
// We need to one-hot encode the Y value since we are doing classification using
// softmax.
trainY1C := Utils.ToOneHot(trainY0C, 3);  // Three classes

trainXC := Tensor.R4.MakeTensor([0, featureCount], trainX0C, wi:=2);
trainYC:= Tensor.R4.MakeTensor([0, 3], trainY1C, wi:=2);
OUTPUT(trainXC, NAMED('trainXC'));
OUTPUT(trainYC, NAMED('trainYC'));

testX0C := NORMALIZE(testC, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
testY0C := NORMALIZE(testC, 3, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.Y));
// We need to one-hot enncode the Y value since we are doing classification using
// softmax.
testY1C := Utils.ToOneHot(testY0C, 3);  // Three classes

testXC := Tensor.R4.MakeTensor([0, featureCount], testX0C, wi:=2);
testYC:= Tensor.R4.MakeTensor([0, 3], testY1C, wi:=2);

// New we can create the combined Train and Test data by adding (i.e. concatenating)
// the Regression and Classification Tensors.  Note that we use wi = 1 (default) for the first
// (i.e. Regression tensor), and wi = 2 for the second (i.e. Classification) tensors
// Classification Tensors.  Thus we have a Tensor List with 2 tensors for each training
// and test input.
trainX := trainXR + trainXC;
trainY := trainYR + trainYC;
OUTPUT(trainX, NAMED('trainX'));
OUTPUT(trainY, NAMED('trainY'));
testX := testXR + testXC;
testY := testYR + testYC;
OUTPUT(testX, NAMED('testX'));
OUTPUT(testY, NAMED('testY'));
// Create a functional model with two inputs, and two outputs.  This is a combination
// of a regression model and a classification model.  The two models are combined into
// a single functional model.  The attributes ldef1 and ldef2 illustrate what the
// individual models would look like using a sequential model definition.
// This allows the reader to see how to map from sequential models to
// a functional model.
// Note that the two models are not connected internally. This is not the normal
// way to use functional models, but is simple and does allow us to test multiple
// inputs and outputs as well as the functional definition process.

// ldef1 is the sequential model definition for the Regression model (for reference)
ldef1 := ['''layers.Dense(256, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.Dense(1, activation=None)'''];
// ldef2 is the sequential model definition for the Classification model (for reference)
ldef2 := ['''layers.Dense(16, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(16, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

// fldef is the Functional model definition that combines the two above models
// The first field is the name of that layer.  The second is the definition of the layer.
// The third field is a list of predecessor layer names for this layer.  The
// Functional model defines a Directed Acyclic Graph (DAG), which is stitched together
// using the predecessor list.   If there were Concatenation layers, they would
// list multiple predecessors.  Note that Input layers have no predecessors.
fldef := DATASET([{'input1', '''layers.Input(shape=(5,))''', []},  // Regression Input
                {'d1', '''layers.Dense(256, activation='tanh')''', ['input1']}, // Regression Hidden 1
                {'d2', '''layers.Dense(256, activation='relu')''', ['d1']},   // Regression Hidden 2
                {'output1', '''layers.Dense(1, activation=None)''', ['d2']}, // Regression Output
                {'input2', '''layers.Input(shape=(5,))''', []}, // Classification Input
                {'d3', '''layers.Dense(16, activation='tanh', input_shape=(5,))''',['input2']}, // Classification Hidden 1
                {'d4', '''layers.Dense(16, activation='relu')''',['d3']}, // Classification Hidden 2
                {'output2', '''layers.Dense(3, activation='softmax')''', ['d4']}], // Classification Output
            FuncLayerDef);

// compileDef defines the compile line to use for compiling the defined model.
// Note that 'model.' is implied, and should not be included in the compile line.
// We define two losses, one for each output.  lossWeights allows us to weight one
// loss stronger than the other if desired.
compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.categorical_crossentropy],
              metrics=[])
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
//   fldef defines the functional model
//   inputs lists the input layer names
//   outputs lists the output layer names
//   compileDef contains the Keras compile statement.
mod := GNNI.DefineFuncModel(s, fldef, ['input1', 'input2'], ['output1', 'output2'], compileDef);
// GetWeights returns the initialized weights that have been synchronized across all nodes.
wts := GNNI.GetWeights(mod);

OUTPUT(wts, NAMED('InitWeights'));

// Fit trains the models, given training X and Y data.  BatchSize is not the Keras batchSize,
// but defines how many records are processed on each node before synchronizing the weights
mod2 := GNNI.Fit(mod, trainX, trainY, batchSize := batchSize, numEpochs := numEpochs);

OUTPUT(mod2, NAMED('mod2'));

// GetLoss returns the average loss for the final training epoch
losses := GNNI.GetLoss(mod2);

OUTPUT(losses, NAMED('Losses'));

// EvaluateMod computes the loss, as well as any other metrics that were defined in the Keras
// compile line.
metrics := GNNI.EvaluateMod(mod2, testX, testY);

OUTPUT(metrics, NAMED('metrics'));

// Predict computes the neural network output given a set of inputs.
preds := GNNI.Predict(mod2, testX);

// Note that the Tensor is a packed structure of Tensor slices.  GetData()
// extracts the data into a sparse cell-based form -- each record represents
// one Tensor cell.  See Tensor.R4.TensData.
testYDatR := Tensor.R4.GetData(testY(wi=1));
testYDatC0 := Tensor.R4.GetData(testY(wi=2));
predDatR := Tensor.R4.GetData(preds(wi=1));
predDatC0 := Tensor.R4.GetData(preds(wi=2));
testYDatC := Utils.FromOneHot(testYDatC0);
predDatC := Utils.FromOneHot(predDatC0);
OUTPUT(SORT(testYDatR, indexes), ALL, NAMED('testDatR'));
OUTPUT(SORT(testYDatC, indexes), ALL, NAMED('testDatC'));
OUTPUT(preds, NAMED('predictions'));
OUTPUT(SORT(predDatR, indexes), ALL, NAMED('predDatR'));
OUTPUT(SORT(predDatC, indexes), ALL, NAMED('predDatC'));

// Here  we are comparing the expected values (testYDat) with the predicted values (predDat)
// and calculating the error and squared error to validate the process
cmpR := JOIN(testYDatR, predDatR, LEFT.indexes[1] = RIGHT.indexes[1], TRANSFORM({SET OF REAL indexes,
                                              REAL pred, REAL actual, REAL error, REAL sqError},
                  SELF.indexes := LEFT.indexes, SELF.pred := RIGHT.value, SELF.actual := LEFT.value,
                  SELF.error := ABS(SELF.actual - SELF.pred),
                  SELF.sqError := SELF.error * SELF.error), LEFT OUTER, LOCAL);

cmpC := JOIN(testYDatC, predDatC, LEFT.indexes[1] = RIGHT.indexes[1], TRANSFORM({SET OF REAL indexes,
                                              REAL pred, REAL actual, BOOLEAN correct},
                  SELF.indexes := LEFT.indexes, SELF.pred := RIGHT.value, SELF.actual := LEFT.value,
                  SELF.correct := SELF.pred = SELF.actual), LEFT OUTER, LOCAL);

OUTPUT(SORT(cmpR, indexes), ALL, NAMED('predcompareR'));
OUTPUT(SORT(cmpC, indexes), ALL, NAMED('predcompareC'));

accuracyR := AVE(cmpR, sqError);
accuracyC := COUNT(cmpC(correct=TRUE)) / COUNT(cmpC);
OUTPUT(accuracyR, NAMED('accuracyR'));
OUTPUT(accuracyC, NAMED('accuracyC'));