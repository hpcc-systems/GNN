
/*
Test results
Environments:
Number of nodes: 1
CPU: Intel Core i7 13700K
GPU: NVIDIA GeForce RTX 4090
Memory: 64GB
Tensorflow version = 2.12
Use GPU

1. Test 1
Start Time = 102243
End Time   = 125855
epoch      = 5
Losses     = 1.857115875701515



2. Test 2
Start Time = 131344
End Time   = 191020
epoch      = 10
Losses     = 0.535207177911486

2. Test 3
Start Time = 33526
End Time   = 123024
epoch      = 15
Losses     = 0.1818407013708231


3. Test 3
GPU        = unavailable
# node     = 1
Start Time = 72030
End Time   = 72738
Losses     = 0.05964114850697418
Metrics    = 
  Loss     = 0.03408002480864525
  Acc      = 0.9901000261306763

4. Test 4
GPU        = unavailable
# node     = 1
Start Time = 73255
End Time   = 73545
Losses     = 0.06493064113116513
Metrics    = 
  Loss     = 0.0379653237760067
  Acc      = 0.9911999702453613
*/

IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS Int;
IMPORT STD;

kString := iTypes.kString;
kStrType := iTypes.kStrType;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

// Test parameters
batchSize := 1000;
numEpochs := 1;
trainToLoss := .0001;
bsr := .25; // BatchSizeReduction.  1 = no reduction.  .25 = reduction to 25% of original.
lrr := 1.0;  // Learning Rate Reduction.  1 = no reduction.  .1 = reduction to 10 percent of original.

// Test GPU
GPU := GNNI.isGPUAvailable();
OUTPUT(GPU, NAMED('isGPUAvailable'));

// Get training data
SET OF REAL get_train_X() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  cifar100 = tf.keras.datasets.cifar100
  (x_train, y_train), (x_test, y_test) = cifar100.load_data()
  x_train = x_train[:2]
  x_train = x_train*1.0/255
  return x_train.flatten().tolist()
ENDEMBED;

SET OF REAL get_train_Y() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  cifar100 = tf.keras.datasets.cifar100
  (x_train, y_train), (x_test, y_test) = cifar100.load_data()
  y_train = y_train[:2]
  y_one_hot = np.eye(100)[y_train.flatten()]
  res = y_one_hot.flatten().tolist()
  return y_one_hot.flatten().tolist()
ENDEMBED;

train_X := get_train_X();
train_Y := get_train_Y();

t1Rec := RECORD
  REAL value;
END;

intpuRec := RECORD
  UNSIGNED8 id;
  REAL value;
END;

x1 := DATASET(train_X, t1Rec);
y1 := DATASET(train_Y, t1Rec);
x2 := PROJECT(x1, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));
y2 := PROJECT(y1, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));

x3 := PROJECT(x2, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/(32*32*3)) + 1, TRUNCATE(
  LEFT.id%(32*32*3)/(32*3)) + 1, TRUNCATE(LEFT.id%(32*3)/3) + 1, LEFT.id%3 + 1], SELF.value := LEFT.value));
y3 := PROJECT(y2, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/100) + 1, LEFT.id%100 + 1], 
  SELF.value := LEFT.value));

x := Tensor.R4.MakeTensor([0,32,32,3], x3);
y := Tensor.R4.MakeTensor([0, 100], y3);

// Define model

s := GNNI.GetSession(1);

ldef := ['''layers.UpSampling2D(size=(7, 7), interpolation='bilinear', input_shape=(32, 32, 3))''',
          '''applications.resnet50.ResNet50(include_top = False, weights = "imagenet")''',
          '''layers.GlobalAveragePooling2D()''',
          '''layers.Dropout(0.25)''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.BatchNormalization()''',
          '''layers.Dense(100, activation='softmax')'''];

compileDef := '''compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), 
                loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
              ''';

mod := GNNI.DefineModel(s, ldef, compileDef);
mod_summary := GNNI.getSummary(mod);

OUTPUT(mod_summary, NAMED('mod_summary'));

// Train model
mod2 := GNNI.Fit(mod, x, y, batchSize := batchSize, numEpochs := numEpochs,
                      trainToLoss := trainToLoss, learningRateReduction := lrr,
                      batchSizeReduction := bsr);
losses := GNNI.GetLoss(mod2);


// Evaluate this model

// Get the testing set
SET OF REAL get_test_X() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  cifar100 = tf.keras.datasets.cifar100
  (x_train, y_train), (x_test, y_test) = cifar100.load_data()
  x_test = x_test[:1000]
  x_test = x_test*1.0/255
  return x_test.flatten().tolist()
ENDEMBED;

SET OF REAL get_test_Y() := EMBED(Python)
  import tensorflow as tf
  import numpy as np
  cifar100 = tf.keras.datasets.cifar100
  (x_train, y_train), (x_test, y_test) = cifar100.load_data()
  y_test = y_test[:1000]
  y_one_hot = np.eye(100)[y_test.flatten()]
  res = y_one_hot.flatten().tolist()
  return y_one_hot.flatten().tolist()
ENDEMBED;

test_X := get_test_X();
test_Y := get_test_Y();


x1_test := DATASET(test_X, t1Rec);
y1_test := DATASET(test_Y, t1Rec);
x2_test := PROJECT(x1_test, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));
y2_test := PROJECT(y1_test, TRANSFORM(intpuRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));


x3_test := PROJECT(x2_test, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/(32*32*3)) + 1, TRUNCATE(
  LEFT.id%(32*32*3)/(32*3)) + 1, TRUNCATE(LEFT.id%(32*3)/3) + 1, LEFT.id%3 + 1], SELF.value := LEFT.value));
y3_test := PROJECT(y2_test, TRANSFORM(TensData, SELF.indexes := [TRUNCATE(LEFT.id/100) + 1, LEFT.id%100 + 1], 
  SELF.value := LEFT.value));


x_test := Tensor.R4.MakeTensor([0,32,32,3], x3_test);
y_test := Tensor.R4.MakeTensor([0, 100], y3_test);
OUTPUT(y_test,NAMED('y_test'));
OUTPUT(y,NAMED('y'));

metrics := GNNI.EvaluateMod(mod2, x_test, y_test);
preds := GNNI.Predict(mod2, x_test);

// OUTPUT results
ORDERED([OUTPUT(STD.Date.CurrentTime(TRUE), NAMED('startTime')), 
  OUTPUT(mod2, NAMED('mod2')),
  OUTPUT(STD.Date.CurrentTime(TRUE), NAMED('endTime')),
  OUTPUT(losses, NAMED('losses')),
  OUTPUT(metrics, NAMED('metrics')),
  OUTPUT(preds, NAMED('preds'))]);
