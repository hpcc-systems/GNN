/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  *This test uses the Human Activity Recognition (HAR) dataset of gesture time series to classify the activity
  * being performed.  For information on this dataset, see:  GNN/Test/Datasets/HAR/Raw/README.txt.
  * This test specifically evaluates the gestures using an LSTM Recurrent Neural Network.
  * Before running these HAR tests, the two datasets under HAR/Processed should be
  * uploaded to the landing zone, and sprayed to the Thor cluster.
  */
IMPORT $.HARDataset as H;
IMPORT Python3 AS Python;
IMPORT $.^.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT GNN.Utils;
IMPORT Std.System.Thorlib;

t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

// *** Configuration Params
// Number of Training records to use.
trainCount := 5000;
// Number of Test records to use.
testCount := 1000;
// Number of Epochs to run
numEpochs := 20;
// Records per batch per node
batchSize := 128;
// *** End Config Params


// Temporary format to hold each observation (i.e. all 72 features)
trainRec := RECORD
  UNSIGNED id;
  SET OF DECIMAL5_4 x;
  UNSIGNED2 y;
END;
// Output the number of train and test records in the dataset.  This is the highest sensible number to use
// for trainCount (above).
OUTPUT(COUNT(H.train), NAMED('MaxTrainingSize'));
OUTPUT(COUNT(H.test), NAMED('MaxTestSize'));

// Filter the datasets to only trainCount and testCount records.
tr0 := H.train[.. trainCount];
te0 := H.test[.. testCount];

// Assign record ids sequentially.  Note.  Ids and features must be sequentially
// numbered from 1 - max.
tr := PROJECT(tr0, TRANSFORM(RECORDOF(LEFT), SELF.id := COUNTER, SELF := LEFT));
te := PROJECT(te0, TRANSFORM(RECORDOF(LEFT), SELF.id := COUNTER, SELF := LEFT));

// Uncomment the following 2 lines to see the data at this stage.
// OUTPUT(SORT(tr, id), ALL, NAMED('tr'));
// OUTPUT(te, ALL, NAMED('te'));

// Convert each 72 feature sample from explicit feature names to a list of features.
trainRec convert(RECORDOF(tr) L, INTEGER C) := TRANSFORM
    SELF.x := [L.one_0, L.two_0, L.three_0, L.four_0, L.five_0, L.six_0, L.seven_0, L.eight_0, L.nine_0, L.one_1, L.two_1, L.three_1, L.four_1, L.five_1, L.six_1, L.seven_1, L.eight_1, L.nine_1, L.one_2, L.two_2, L.three_2, L.four_2, L.five_2, L.six_2, L.seven_2, L.eight_2, L.nine_2, L.one_3, L.two_3, L.three_3, L.four_3, L.five_3, L.six_3, L.seven_3, L.eight_3, L.nine_3, L.one_4, L.two_4, L.three_4, L.four_4, L.five_4, L.six_4, L.seven_4, L.eight_4, L.nine_4, L.one_5, L.two_5, L.three_5, L.four_5, L.five_5, L.six_5, L.seven_5, L.eight_5, L.nine_5, L.one_6, L.two_6, L.three_6, L.four_6, L.five_6, L.six_6, L.seven_6, L.eight_6, L.nine_6, L.one_7, L.two_7, L.three_7, L.four_7, L.five_7, L.six_7, L.seven_7, L.eight_7, L.nine_7];
    SELF.y := L.class_;
    SELF.id := L.id;
END;

train0 := PROJECT(tr,convert(LEFT,COUNTER));
test0 := PROJECT(te,convert(LEFT,COUNTER));

// Create the training Tensors
// Create 72 tensor data cells per record (i.e. 8 timesteps with 9 features each)
trainX0 := NORMALIZE(train0, 72, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1)/9 + 1,(COUNTER-1)%9 + 1],
                            SELF.value := LEFT.x[COUNTER]));
// Create 6 tensor data cells per Y value.  These are 1-hot encoded with the target class 1 and all other classes 0.
// Note that cells with 0 value can be skipped since TensData is a sparse format.  This should slightly improve
// performance.  We could also use Utils.ToOneHot(...), but we do it this way to illustrate how the OneHot tensor is
// formed.
trainY0 := NORMALIZE(train0, 6, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := IF(COUNTER = LEFT.y + 1,1,SKIP)));
// Create the training tensors.  Note that the dimensions for the independent (X) data are [numRecords, numTimesteps, numFeatures]
// Note the 0 in the numRecords dimension means any number of records.  This should always be zero for X and Y data.
trainX := Tensor.R4.MakeTensor([0, 8, 9], trainX0);
// Dimensions for the dependent (Y) data are [numRecords, numClasses]
trainY:= Tensor.R4.MakeTensor([0, 6], trainY0);

// These lines are used to output the tensors without the full data, so that we can see how the tensor slices are stored
// (for debugging only)
trainXtmp := PROJECT(trainX, {t_Tensor - sparsedata - densedata});
trainYtmp := PROJECT(trainY, {t_Tensor - sparsedata - densedata});
//OUTPUT(trainXtmp, NAMED('TrainX'));
//OUTPUT(trainYtmp, NAMED('TrainY'));

// Now create the testing Tensors (same as above, but using the test dataset).  This time we use
// Utils.ToOneHot(...) for the Y data to illustrate that technique.
testX0 := NORMALIZE(test0, 72, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, (COUNTER-1)/9 + 1,(COUNTER-1)%9 + 1],
                            SELF.value := LEFT.x[COUNTER]));

testY0tmp := PROJECT(test0, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, 1],
                            SELF.value := LEFT.y));
testY0 := Utils.ToOneHot(testY0tmp, 6);  // 6 is the number of features

testX := Tensor.R4.MakeTensor([0, 8, 9], testX0);
testY := Tensor.R4.MakeTensor([0, 6], testY0);

// Definition of the Keras model.  Note that the first layer needs input_shape defined (per Keras).
// This is the shape of a single observation.  The 6 in the final softmax layer is the number
// of output classes.
// Note that labels: "tf" (for tensorflow) and "layers" (a shortcut for tf.keras.layers) are
// available for use in these layer definitions as well as in compileDef (below).
ldef :=['''layers.LSTM(64, input_shape=(8,9))''',
                '''layers.Dropout(0.5)''',
                '''layers.Dense(32, activation='relu')''',
                '''layers.Dense(6, activation='softmax')'''];

// Specification of the Keras compile line.  Note that in Keras, you would do: model.compile(compileDef).
// For GNN, the model parameter is implied, ahd should not be included in the compile line.  Amy
// Keras supported compile parameters can be used in this line.
compileDef := '''compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])''';

// Get the GNNI session token.  This is used as input to DefineModel below.
s := GNNI.GetSession();
// Define the model.  The returned model token is used in subsequent calls.
mod := GNNI.DefineModel(s, ldef, compileDef);
// Get the weights
wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));
// Train the model.  See notes on batchSize and numEpochs in the GNNI documentation.  Note that
// a second model token is returned here.  It is important that this second token be used
// for subsequent calls as it references the trained model, whereas mod references the pre-training
// model.
mod2 := GNNI.Fit(mod, trainX, trainY, batchSize := batchSize, numEpochs := numEpochs);
// Evaluate the model.
// Returns a set of metrics as defined by compileDef above.
metrics := GNNI.EvaluateMod(mod2, testX, testY);
OUTPUT(metrics, NAMED('Metrics'));

// Use the model for predictions.  Note that the predictions are in the form of probabilities for
// each output class.  To choose the class with the highest probability, we call
// Utils.Probabilities2Class(...) below.
preds := GNNI.Predict(mod2, testX);
testYDat := Tensor.R4.GetData(testY);
predDat := Tensor.R4.GetData(preds);
OUTPUT(SORT(predDat, indexes), ALL, NAMED('PredDat'));

predDatClass := Utils.Probabilities2Class(predDat);
testYDatClass := Utils.Probabilities2Class(testYDat);
OUTPUT(SORT(predDatClass, indexes), ALL, NAMED('PredDatClass'));
OUTPUT(SORT(testYDatClass, indexes), ALL, NAMED('TestYDatClass'));

// Evaluate the predictions
cmp := JOIN(predDatClass, testYDatClass,
                        LEFT.indexes[1] = RIGHT.indexes[1],
                        TRANSFORM({UNSIGNED id, UNSIGNED pred, UNSIGNED actual,
                                      BOOLEAN correct},
                                      SELF.id := LEFT.indexes[1],
                                      SELF.pred := LEFT.value,
                                      SELF.actual := RIGHT.value,
                                      SELF.correct := SELF.pred = SELF.actual),
                          LOCAL);
OUTPUT(SORT(cmp, id), ALL, NAMED('PredCompare'));

acc := COUNT(cmp(correct = TRUE)) / COUNT(cmp);

OUTPUT(acc, NAMED('Accuracy'));