/*
About this test:
    Some basic unit tests for Functional API.
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
FuncLayerDef := Types.FuncLayerDef;

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

compileDef := '''compile(optimizer=tf.keras.optimizers.SGD(.05),
              loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.categorical_crossentropy],
              metrics=[])
              ''';

OUTPUT(fldef, NAMED('fldef')); 
OUTPUT(compileDef, NAMED('compileDef')); 

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

modSummary := GNNI.getSummary(mod);
OUTPUT(modSummary, NAMED('modSummary'));

// GetWeights returns the initialized weights that have been synchronized across all nodes.
wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));

NewWeights := PROJECT(wts, TRANSFORM(t_Tensor, SELF := LEFT));
OUTPUT(NewWeights, NAMED('NewWeights'));
mod2 := GNNI.SetWeights(mod, NewWeights);
wts2 := GNNI.GetWeights(mod2);
OUTPUT(wts2, NAMED('SetWeights'));

mod2FullModel := GNNI.getModel(mod2);
OUTPUT(mod2FullModel, NAMED('mod2FullModel'));

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

// Test with pre-trained model
// ldef3 is the sequential model definition (for reference)
ldef3 := ['''layers.UpSampling2D(size=(7, 7), interpolation='bilinear', input_shape=(32, 32, 3))''',
          '''applications.resnet50.ResNet50(include_top = False, weights = "imagenet")''',
          '''layers.GlobalAveragePooling2D()''',
          '''layers.Dropout(0.25)''',
          '''layers.Dense(256, activation='relu')''',
          '''layers.BatchNormalization()''',
          '''layers.Dense(100, activation='softmax')'''];

compileDef3 := '''compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), 
                loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
              ''';

fldef3 := DATASET([{'input', '''layers.Input(shape=(32, 32, 3))''', []},  // Input
                {'UpSampling', '''layers.UpSampling2D(size=(7, 7), interpolation='bilinear')''', ['input']},  // UpSampling Layer
                {'ResNet50', '''applications.resnet50.ResNet50(include_top = False, weights = "imagenet")''', ['UpSampling']}, // ResNet50
                {'Pooling', '''layers.GlobalAveragePooling2D()''', ['ResNet50']},   // Pooling Layer
                {'Dropout', '''layers.Dropout(0.25)''', ['Pooling']},   // Dropout Layer
                {'Dense1', '''layers.Dense(256, activation='relu')''', ['Dropout']},   // Dense Layer
                {'Normalization', '''layers.BatchNormalization()''', ['Dense1']},   // Normalization Layer
                {'output', '''layers.Dense(100, activation='softmax')''', ['Normalization']}], // Output
            FuncLayerDef);

mod5 := GNNI.DefineFuncModel(s, fldef3, ['input'], ['output'], compileDef3);
mod5_summary := GNNI.getSummary(mod5);
OUTPUT(mod5_summary, NAMED('Functional_with_pre_trained'));