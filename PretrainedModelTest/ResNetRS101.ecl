/*
About this test:
  Test the usability of Pre-trained Model ResNetRS101.
  Reference: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_rs/ResNetRS101
  Input shape = (224, 224, 3) 

Results:

class                   probability
tusker	                0.6600980162620544
African_elephant	      0.1865376234054565
Indian_elephant	        0.01173316035419703

*/

IMPORT Python3 AS Python;
IMPORT $.^ AS GNN;
IMPORT GNN.GNNI;
IMPORT GNN.Tensor;
IMPORT GNN.Internal AS int;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT Std.System.Thorlib;
IMPORT STD;

kString := iTypes.kString;
kStrType := iTypes.kStrType;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

// load the test data, an image of a elephant
imageRecord := RECORD
  STRING filename;
  DATA   image;   
       //first 4 bytes contain the length of the image data
  UNSIGNED8  RecPos{virtual(fileposition)};
END;

imageData := DATASET('~le::elephant',imageRecord,FLAT);
OUTPUT(imageData, NAMED('elephant'));

result := (STRING)(imageData[1].image);

SET OF INTEGER hexToNparry(DATA byte_array):= EMBED(Python)
  from PIL import Image
  import numpy as np
  import io
  try:
    import tensorflow as tf # V2.x
  except:
    assert 1 == 0, 'tensorflow not found'
  bytes_data = bytes(byte_array)
  image = Image.open(io.BytesIO(bytes_data))
  image = image.resize((224,224))
  I_array = np.array(image)
  I_array = tf.keras.applications.resnet_rs.preprocess_input(I_array)
  return I_array.flatten().tolist()
ENDEMBED;

valueRec := RECORD
  INTEGER value;
END;

idValueRec := RECORD
  UNSIGNED8 id;
  INTEGER value;
END;

imageNpArray := hexToNparry(imageData[1].image);
x1 := DATASET(imageNpArray, valueRec);
x2 := PROJECT(x1, TRANSFORM(idValueRec, SELF.id := COUNTER - 1, SELF.value := LEFT.value));
x3 := PROJECT(x2, TRANSFORM(TensData, SELF.indexes := [1, TRUNCATE(LEFT.id/(224*3)) + 1, TRUNCATE(LEFT.id/3)%224 + 1, LEFT.id%3 + 1], SELF.value := LEFT.value));
x := Tensor.R4.MakeTensor([0,224,224,3], x3);

// load the model
s := GNNI.GetSession(1);
ldef := ['''applications.resnet_rs.ResNetRS101(weights = "imagenet")'''];
mod := GNNI.DefineModel(s, ldef);

// Predict 
preds_tens := GNNI.Predict(mod, x);
preds := Tensor.R4.GetData(preds_tens);

predictRes := RECORD
  STRING class;
  REAL4 probability;
END;

// decode predictions
DATASET(predictRes) decodePredictions(DATASET(TensData) preds, INTEGER topK = 3) := EMBED(Python)
  try:
    from tensorflow.keras.applications.resnet_rs import decode_predictions
  except:
    assert 1 == 0, 'tensorflow not found'
  import numpy as np
  predsNp = np.zeros((1, 1000))
  for pred in preds:
    predsNp[0, pred[0][1]-1] = pred[1]
  res = decode_predictions(predsNp, top=topK)[0]
  ret = []
  for i in range(topK):
    ret.append((res[i][1], res[i][2]))
  return ret
ENDEMBED;

OUTPUT(decodePredictions(preds), NAMED('predictions'));
