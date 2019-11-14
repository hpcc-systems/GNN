/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Internal AS int;
IMPORT Std.System.Thorlib;

kString := iTypes.kString;
initParms := iTypes.initParms;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
nodeId := Thorlib.node();
nNodes := Thorlib.nodes();
// Prepare training data
RAND_MAX := POWER(2,32) -1;
//trainCount := 10000;
trainCount := 1000;
featureCount := 5;
trainRec := RECORD
  UNSIGNED8 id;
  SET OF REAL x;
  REAL4 y;
END;
REAL4 targetFunc(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt := .5 * POWER(x1, 4) - .4 * POWER(x2, 3) + .3 * POWER(x3,2) - .2 * x4 + .1 * x5;
  RETURN rslt;
END;
train0 := DATASET(trainCount, TRANSFORM(trainRec,
                      SELF.id := COUNTER,
                      SELF.x := [(RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5,
                                  (RANDOM() % RAND_MAX) / RAND_MAX -.5],
                      SELF.y := targetFunc(SELF.x[1], SELF.x[2], SELF.x[3], SELF.x[4], SELF.x[5]))
                      );
OUTPUT(train0, NAMED('trainData'));


trainX0 := NORMALIZE(train0, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
trainY0 := NORMALIZE(train0, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));
X := Tensor.R4.MakeTensor([0, featureCount], trainX0);
Y:= Tensor.R4.MakeTensor([0, 1], trainY0);
OUTPUT(X, NAMED('X'));
OUTPUT(Y, NAMED('Y'));

eX1 := int.TensExtract(X, 1, 10);
eX2 := int.TensExtract(X, 11, 10);
eX3 := int.TensExtract(X, 330, 40);

OUTPUT(eX1, NAMED('eX1'));
OUTPUT(eX2, NAMED('eX2'));
OUTPUT(eX3, NAMED('eX3'));