IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.GNNI;
IMPORT STD.Date;
IMPORT GNN.Tensor;
IMPORT GNN.Internal AS int;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT Std.System.Thorlib;

// blob data created using https://github.com/hpcc-systems/GPU-Deep-Learning/blob/master/Datasets/data_files/Dataset%20Preprocessing.ipynb
// and imported with blob fix length (785) type

// hyperparam
batchSize := 128;
numEpochs := 5;
effNodes := 1;

mnist_data_type := RECORD
	 INTEGER1 label;
	 DATA784 image;
END;


mnist_data_type_withid_x := RECORD
	 UNSIGNED id;   
	 SET Image;
END;

mnist_data_type_withid_y := RECORD
	 UNSIGNED id;     
     INTEGER1 label;
END;

// how to convert hex to integer
train0 := DATASET('~mnist::train', mnist_data_type, THOR);
test0 := DATASET('~mnist::test', mnist_data_type, THOR);
trainBig0 := DATASET('~mnist::big::train', mnist_data_type, THOR);
testBig0 := DATASET('~mnist::big::test', mnist_data_type, THOR);


SET byte2int(DATA784 image) := EMBED(Python)
    import numpy as np
    return np.asarray(image, dtype='B').astype('int').tolist()
ENDEMBED;

trainX1 := PROJECT(train0, TRANSFORM(mnist_data_type_withid_x, SELF.image:=byte2int(LEFT.image), SELF.id := COUNTER, SELF := LEFT));
trainY1 := PROJECT(train0, TRANSFORM(mnist_data_type_withid_y, SELF.label:=LEFT.label, SELF.id := COUNTER, SELF:= LEFT));
//output(trainX1);

trainX2 := NORMALIZE(trainX1, 784, TRANSFORM(Tensor.R4.TensData,
                            SELF.indexes := [LEFT.id, ((counter-1) div 28) + 1, ((counter-1) % 28) +1],
                            SELF.value := ( (REAL) LEFT.image[counter])/127.5 -1));
output(trainX2, named('trainX2'));
trainY2 := NORMALIZE(trainY1, 10, TRANSFORM(Tensor.R4.TensData,
                            SELF.indexes := [LEFT.id, counter],
                            SELF.value := IF(COUNTER = LEFT.label + 1,1,SKIP)));

trainX3 := Tensor.R4.MakeTensor([0,28, 28], trainX2);
trainY3 := Tensor.R4.MakeTensor([0, 10], trainY2);
// output(trainX, named('trainX'));
//output(trainY3, named('trainY3'));

testX1 := PROJECT(test0, TRANSFORM(mnist_data_type_withid_x, SELF.image:=byte2int(LEFT.image), SELF.id := COUNTER, SELF := LEFT));
testY1 := PROJECT(test0, TRANSFORM(mnist_data_type_withid_y, SELF.label:=LEFT.label, SELF.id := COUNTER, SELF:= LEFT));



testX2 := NORMALIZE(testX1, 784, TRANSFORM(Tensor.R4.TensData,
                            SELF.indexes := [LEFT.id, ((counter-1) div 28) + 1, ((counter-1) % 28) +1],
                            SELF.value := ( (REAL) LEFT.image[counter])/127.5 -1));

testY2 := NORMALIZE(testY1, 10, TRANSFORM(Tensor.R4.TensData,
                            SELF.indexes := [LEFT.id, counter],
                            SELF.value := IF(COUNTER = LEFT.label + 1,1,SKIP)));
testX3 := Tensor.R4.MakeTensor([0,28, 28], testX2);
testY3 := Tensor.R4.MakeTensor([0, 10], testY2);


s := GNNI.GetSession();

ldef := [ '''tf.keras.layers.Flatten(input_shape=(28,28))''',
          '''layers.Dense(512, activation='tanh')''',
          '''layers.Dense(512, activation='relu')''',
          '''layers.Dense(10, activation='softmax')'''];

compileDef := '''compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])''';

mod := GNNI.DefineModel(s, ldef, compileDef);

wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));


startTime := Date.CurrentSeconds(true):CHECKPOINT('startTime');

mod2 := GNNI.Fit(
    mod, trainX3, trainY3, 
    batchSize := batchSize, 
    numEpochs := numEpochs,
    limitNodes := 0
    );

endTime := Date.CurrentSeconds(true);
// OUTPUT(mod3, NAMED('mod3'));

#WORKUNIT('name', 'mnist_small_n__0');

SEQUENTIAL(
  OUTPUT(Date.SecondsToString(startTime, '%H:%M:%S'), NAMED('StartTime')),
  OUTPUT(mod2, NAMED('mod2')),
  OUTPUT(Date.SecondsToString(endTime, '%H:%M:%S'), NAMED('EndTime')),
  OUTPUT(endtime-starttime, NAMED('TimeTaken'))
);
/*
losses := GNNI.GetLoss(mod2);
output(losses, NAMED('LOSSES'));
metrics := GNNI.EvaluateMod(mod2, testX3, testY3);
preds := GNNI.Predict(mod2, testX3);

OUTPUT(testY3, ALL, NAMED('testDat'));
OUTPUT(preds, NAMED('predictions'));

// ldef := [
// 	'''layers.Dense(256, activation='tanh', input_shape=(5,))''',
// 	'''layers.Dense(256, activation='relu')''',
//     '''layers.Dense(1, activation=None)'''];

// compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),                  loss=tf.keras.losses.mean_squared_error,                  metrics=[tf.keras.metrics.mean_squared_error]) ''';

// mod := GNNI.DefineModel(s, ldef, compileDef);

// mod2 := GNNI.Fit(mod, trainX, trainY, batchSize := 128, numEpochs := 5);
// initialWts := GNNI.GetWeights(mod);  // The untrained model weights
// trainedWts := GNNI.GetWeights(mod2);  // The trained model weights
// metrics := GNNI.EvaluateMod(mod2, testX, testY);
// preds := GNNI.Predict(mod2, predX);

*/