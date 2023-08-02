/*
About this test:
    Test the performance of training in TensorFlow 1.x
    Add more neurons in ldef to increase training time.

Test Results:

1. 
Model      = 
ldef := ['''layers.Dense(256, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(1024, activation='relu')''',
          '''layers.Dense(1024, activation='relu')''',
          '''layers.Dense(10240, activation='relu')''',
          '''layers.Dense(1024, activation='relu')''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

Start Time = 33029
End Time   = 40211
loss       = 0.02119083986617625

*/

IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT ML_Core AS mlc;
IMPORT STD;

kString := iTypes.kString;
kStrType := iTypes.kStrType;
NumericField := mlc.Types.NumericField;
t_Tensor := Tensor.R4.t_Tensor;

// Prepare training data
RAND_MAX := POWER(2,32) -1;

// Test parameters
trainCount := 10000;
testCount := 100;
featureCount := 5;
classCount := 3;
numEpochs := 10;
batchSize := 128;

// Add more neurons to increase training time
ldef := ['''layers.Dense(256, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(1024, activation='relu')''',
          '''layers.Dense(10240, activation='relu')''',
          '''layers.Dense(10240, activation='relu')''',
          '''layers.Dense(1024, activation='relu')''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
              ''';

s := GNNI.GetSession(1);
mod := GNNI.DefineModel(s, ldef, compileDef);

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

OUTPUT(testX, NAMED('testX'));
OUTPUT(testY, NAMED('testY'));

mod2 := GNNI.FitNF(mod, trainX, trainY, batchSize := batchSize, numEpochs := numEpochs);

losses := GNNI.GetLoss(mod2);
metrics := GNNI.EvaluateNF(mod2, testX, testY);
preds := GNNI.PredictNF(mod2, testX);

ORDERED(
  OUTPUT(STD.Date.CurrentTime(TRUE), NAMED('start')),
  OUTPUT(mod2, NAMED('mod2')),
  OUTPUT(STD.Date.CurrentTime(TRUE), NAMED('end')),
  PARALLEL(
           OUTPUT(losses, NAMED('losses')),
           OUTPUT(metrics, NAMED('metrics')),
           OUTPUT(testY, ALL, NAMED('testDat')),
           OUTPUT(preds, NAMED('predictions'))
           )
        );
