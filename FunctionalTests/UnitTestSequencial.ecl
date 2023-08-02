/*
About this test:
    Some basic unit tests for sequencial API.
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

// Define layers
ldef := ['''layers.Dense(16, activation='tanh', input_shape=(5,))''',
          '''layers.Dense(16, activation='relu')''',
          '''layers.Dense(3, activation='softmax')'''];

compileDef := '''compile(optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=0.05),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
              ''';
OUTPUT(ldef, NAMED('ldef'));                                             
OUTPUT(compileDef, NAMED('compileDef'));   

s := GNNI.GetSession(0);
GPU := GNNI.isGPUAvailable();
SEQUENTIAL(OUTPUT(s, NAMED('s')),
           OUTPUT(GPU, NAMED('isGPUAvailable')));


mod := GNNI.DefineModel(s, ldef, compileDef);

modSummary := GNNI.getSummary(mod);
OUTPUT(modSummary, NAMED('modSummary'));

wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));

NewWeights := PROJECT(wts, TRANSFORM(RECORDOF(LEFT), SELF.denseData := IF(LEFT.wi = 1, 
                [.5, .5, .5] + LEFT.densedata[4..], LEFT.densedata), SELF := LEFT));

OUTPUT(NewWeights, NAMED('NewWeights'));
mod2 := GNNI.SetWeights(mod, NewWeights);
wts2 := GNNI.GetWeights(mod2);
OUTPUT(wts2, NAMED('SetWeights'));

mod2FullModel := GNNI.getModel(mod2);
OUTPUT(mod2FullModel, NAMED('fullModel'));

mod3 := GNNI.setModel(s, mod2FullModel);

mod3FullModel := GNNI.getModel(mod3);
OUTPUT(IF(mod2FullModel = mod3FullModel, 'Pass', 'Fail'), NAMED('getModel_setModel_Test'));

wts3 := GNNI.GetWeights(mod3);

OUTPUT(IF(wts2 = wts3, 'Pass', 'Fail'), NAMED('getWeight_Test'));

STRING mod2JSON := GNNI.ToJSON(mod2);
STRING mod3JSON := GNNI.ToJSON(mod3);
OUTPUT(IF(mod2JSON = mod3JSON, 'Pass', 'Fail'), NAMED('ToJSON_Test'));

mod4 := GNNI.FromJSON(s, mod3JSON);
STRING mod4JSON := GNNI.ToJSON(mod4);
OUTPUT(IF(mod3JSON = mod4JSON, 'Pass', 'Fail'), NAMED('FromJSON_Test'));

// test with pre-trained model

ldef2 := ['''layers.UpSampling2D(size=(7, 7), interpolation='bilinear', input_shape=(32, 32, 3))''',
          '''applications.resnet50.ResNet50(include_top = False, weights = "imagenet")''',
          '''layers.GlobalAveragePooling2D()''',
          '''layers.Dropout(0.25)''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.BatchNormalization()''',
          '''layers.Dense(100, activation='softmax')'''];

compileDef2 := '''compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), 
                loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
              ''';

mod5 := GNNI.DefineModel(s, ldef2, compileDef2);
mod5_summary := GNNI.getSummary(mod5);

OUTPUT(mod5_summary, NAMED('Sequencial_with_pre_trained'));
