IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.GNNI;
IMPORT GNN.Tensor;
IMPORT GNN.Internal AS int;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT Std.System.Thorlib;

// hyperparam
batchSize := 512;
numEpochs := 5;
effNodes := 0;

mnist_data_type := RECORD
	 INTEGER1 label;
	 DATA784 image;
END;


mnist_data_type_withid := RECORD
	 UNSIGNED id;     
     INTEGER1 label;
	 DATA784 image;
END;

// how to convert hex to integer
train0 := DATASET('~mnist::train', mnist_data_type, THOR);
test0 := DATASET('~mnist::test', mnist_data_type, THOR);
trainBig0 := DATASET('~mnist::big::train', mnist_data_type, THOR);
testBig0 := DATASET('~mnist::big::test', mnist_data_type, THOR);

trainX1 := PROJECT(train0, TRANSFORM(mnist_data_type_withid, SELF.image:=LEFT.image, SELF.id := COUNTER, SELF := LEFT));
trainY1 := PROJECT(train0, TRANSFORM(mnist_data_type_withid, SELF.label:=LEFT.label, SELF.id := COUNTER, SELF:= LEFT));



trainX2 := NORMALIZE(trainX1, 784, TRANSFORM(Tensor.R4.TensData,
                            SELF.indexes := [LEFT.id, ((counter-1) div 28) + 1, ((counter-1) % 28) +1],
                            SELF.value := ( (REAL) (>UNSIGNED1<) LEFT.image[counter])/127.5 -1));

trainY2 := NORMALIZE(trainY1, 1, TRANSFORM(Tensor.R4.TensData,
                            SELF.indexes := [LEFT.id, counter],
                            SELF.value := (REAL)LEFT.label));
                            
                            //(REAL)(UNSIGNED)LEFT.image[counter]));

trainX3 := Tensor.R4.MakeTensor([0,28, 28], trainX2);
trainY3 := Tensor.R4.MakeTensor([0, 1], trainY2);
// output(trainX, named('trainX'));
// output(trainY, named('trainY'));


s := GNNI.GetSession();

ldef := ['''layers.Dense(512, activation='tanh', input_shape=(784,))''',
          '''layers.Dense(512, activation='relu')''',
          '''layers.Dense(10, activation='softmax')'''];

compileDef := '''compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
              ''';
mod := GNNI.DefineModel(s, ldef, compileDef);

wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));


mod2 := GNNI.Fit(mod, trainX3, trainY3, batchSize := batchSize, numEpochs := numEpochs);
OUTPUT(mod2, NAMED('mod2'));

losses := GNNI.GetLoss(mod2);
/*
metrics := GNNI.EvaluateNF(mod2, testX, testY);
preds := GNNI.PredictNF(mod2, testX);

OUTPUT(testY, ALL, NAMED('testDat'));
OUTPUT(preds, NAMED('predictions'));

output(train0)
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