/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Unit tests for the Tensor module
  */
IMPORT Python;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal AS int;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT Std.System.Thorlib;

t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
datCount := 100000;
featureCount := 5;

cmpRec := RECORD
  SET OF UNSIGNED4 indexes;
  REAL4 val1;
  REAL4 val2;
  REAL4 diff;
  BOOLEAN correct;
END;
DATASET(cmpRec) compareDat(DATASET(TensData) td1, DATASET(TensData) td2) := FUNCTION
  epsilon := .000000001;
  rslt := JOIN(td1, td2, LEFT.indexes = RIGHT.indexes, TRANSFORM(cmpRec,
                                                SELF.indexes := LEFT.indexes,
                                                SELF.val1 := LEFT.value,
                                                SELF.val2 := RIGHT.value,
                                                SELF.diff := ABS(LEFT.value - RIGHT.value),
                                                SELF.correct := SELF.diff < epsilon), HASH);
  return SORT(rslt, indexes);
END;

testRec := RECORD
  UNSIGNED8 id;
  SET OF REAL x;
  REAL4 y;
END;

// The target function maps a set of X features into a Y value, which is a polynomial function of X.
REAL4 targetFunc(REAL4 x1, REAL4 x2, REAL4 x3, REAL4 x4, REAL4 x5) := FUNCTION
  rslt := (X1 + X2 + X3 + X4 + x5) / 5.0;
  RETURN rslt;
END;

// Build the training data
test0 := DATASET(datCount, TRANSFORM(testRec,
                      SELF.id := COUNTER,
                      SELF.x := [COUNTER,
                                  COUNTER + 1,
                                  COUNTER + 2,
                                  COUNTER + 3,
                                  COUNTER + 4],
                      SELF.y := 0)
                      );
// Be sure to compute Y in a second step.  Otherewise, the RANDOM() will be executed twice and the Y will be based
// on different values than those assigned to X.  This is an ECL quirk that is not easy to fix.
testDat := PROJECT(test0, TRANSFORM(RECORDOF(LEFT), SELF.y := targetFunc(LEFT.x[1], LEFT.x[2], LEFT.x[3], LEFT.x[4], LEFT.x[5]), SELF := LEFT));
OUTPUT(testDat, NAMED('testData'));

tensDatX0 := NORMALIZE(testDat, featureCount, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.x[COUNTER]));
tensDatY0 := NORMALIZE(testDat, 1, TRANSFORM(TensData,
                            SELF.indexes := [LEFT.id, COUNTER],
                            SELF.value := LEFT.y));
OUTPUT(tensDatX0[..1000], ALL, NAMED('tensDatX0'));
OUTPUT(tensDatY0[..1000], ALL, NAMED('tensDatY0'));
// Form a Tensor from the tensor data.  This packs the data into 'slices' that can contain dense
// or sparse portions of the Tensor.  If the tensor is small, it will fit into a single slice.
// Huge tensors may require many slices.  The slices also contain tensor metadata such as the shape.
// For record oriented data, the first component of the shape should be 0, indicating that it is an
// arbitrary length set of records.
// Make a tensor and then retrieve the data.
tensX0 := Tensor.R4.MakeTensor([0, featureCount], tensDatX0, wi:=1);
tensY0:= Tensor.R4.MakeTensor([0, 1], tensDatY0, wi := 1);
OUTPUT(tensX0, NAMED('tensX0'));
OUTPUT(tensY0, NAMED('tensY0'));

tensDatX1 := Tensor.R4.GetData(tensX0);

OUTPUT(tensDatX1[..1000], ALL,  NAMED('tensDataX1'));

// tensDatX1 should not be the same as tensDatX0
cmp1 := SORT(compareDat(tensDatX0, tensDatX1), indexes[1], indexes[2]);
OUTPUT(cmp1[..1000], ALL, NAMED('cmp1'));
cmpErrCnt1 := COUNT(cmp1(correct = FALSE));
cmpErrs1 := cmp1(correct=FALSE);
OUTPUT(cmpErrs1[..1000], ALL, NAMED('cmpErrs1'));
OUTPUT(cmpErrCnt1, NAMED('cmpErrCnt1'));

// Align the x and y tensors so that they have corresponding indexes and the same distribution
tempY0 := PROJECT(tensY0, TRANSFORM(RECORDOF(LEFT), SELF.wi := 2, SELF := LEFT), LOCAL);
OUTPUT(tempY0, NAMED('tempY0'));
aligned := Tensor.R4.AlignTensorPair(tensX0 + tempY0);
OUTPUT(aligned, NAMED('aligned'));
tensX1 := aligned(wi=1);
tensY1 := PROJECT(aligned(wi=2), TRANSFORM(RECORDOF(LEFT),
                                            SELF.wi := 1, SELF := LEFT), LOCAL);
OUTPUT(tensX1, NAMED('tensX1'));
OUTPUT(tensY1, NAMED('tensY1'));

// Now sample from both X and Y and make sure the results jive
pos := (datCount / 2 / Thorlib.nodes()) - 5;
tensX2 := int.TensExtract(tensX1, pos, 10);
tensY2 := int.TensExtract(tensY1, pos, 10);
OUTPUT(tensX2, NAMED('tensX2'));
OUTPUT(tensY2, NAMED('tensY2'));

recCntX2 := Tensor.R4.GetRecordCount(tensX2);
recCntY2 := Tensor.R4.GetRecordCount(tensY2);
OUTPUT(recCntX2, NAMED('recCountX2'));
OUTPUT(recCntY2, NAMED('recCountY2'));

// Try a sparse tensor based on the same X data, but with only every fourth record
tensDataX3 := tensDatX0(indexes[1] % 4 = 0);
tensX3 := Tensor.R4.MakeTensor([0, featureCount], tensDataX3, wi:=1);
OUTPUT(tensX3, NAMED('tensX3'));
tensDataX4 := Tensor.R4.GetData(tensX3);
cmp2 := SORT(compareDat(tensDataX3, tensDataX4), indexes[1], indexes[2]);
OUTPUT(cmp2[..1000], ALL, NAMED('cmp2'));
cmpErrCnt2 := COUNT(cmp2(correct = FALSE));
cmpErrs2 := cmp2(correct=FALSE);
OUTPUT(cmpErrs2[..1000], ALL, NAMED('cmpErrs2'));
OUTPUT(cmpErrCnt2, NAMED('cmpErrCnt2'));

// Try various combinations of basic tensor creation and get data.

tensDat1 := DATASET([{[1,1], 1.1},
                      {[1,2], 1.2},
                      {[1,3], 1.3},
                      {[2,1], 2.1},
                      {[2,2], 2.2},
                      {[2,3], 2.3},
                      {[3,1], 3.1},
                      {[3,2], 3.2},
                      {[3,3], 3.3}], TensData);
// Now a replicated dense tensor
tens2 := Tensor.R4.MakeTensor([3,3], tensDat1, replicated := TRUE);
OUTPUT(tens2, NAMED('tens2'));

retDat2 := Tensor.R4.getData(tens2);
OUTPUT(retDat2, NAMED('retDat2'));

// With shape[9,9] we get a 'sparse' tensor.
tens3 := Tensor.R4.MakeTensor([9,9], tensDat1);
OUTPUT(tens3, NAMED('tens3'));

retDat3 := Tensor.R4.getData(tens3);

OUTPUT(retDat3, NAMED('retDat3'));

// Now a replicated 'sparse' tensor
tens4 := Tensor.R4.MakeTensor([9,9], tensDat1, replicated := TRUE);
OUTPUT(tens4, NAMED('tens4'));

retDat4 := Tensor.R4.getData(tens4);
OUTPUT(retDat4, NAMED('retDat4'));

// Now a multi-slice tensor.  We create a huge shape, even though lightly populated.
// While we're at it, we test tensors > 2D
tensDat2 := DATASET([{[1,1,1,1], 1.1},
                    {[1,1,1,3], 1.3},
                    {[1,1,2,1], 2.1},
                    {[1,1,2,2], 2.2},
                    {[1000,1000,1,1], 1001.1},
                    {[1000,1000,3,3], 1003.3}], TensData);

tens5 := Tensor.R4.MakeTensor([1000,1000, 3,3], tensDat2, replicated := FALSE);
OUTPUT(tens5, NAMED('tens5'));

retDat5 := Tensor.R4.getData(tens5);
OUTPUT(retDat5, NAMED('retDat5'));

tens6 := Tensor.R4.MakeTensor([1000,1000, 3,3], tensDat2, replicated := TRUE);
OUTPUT(tens6, NAMED('tens6'));

retDat6 := Tensor.R4.getData(tens6);
OUTPUT(retDat6, NAMED('retDat6'));

// Now create a record oriented tensor with a zero first shape term

tensDat3 := DATASET([{[1,1], 1.1},
                    {[1,2], 1.2},
                    {[1,3], 1.3},
                    {[2,1], 2.1},
                    {[2,2], 2.2},
                    {[2,3], 2.3},
                    {[3,1], 3.1},
                    {[3,2], 3.2},
                    {[3,3], 3.3}], TensData);

tens7 := Tensor.R4.MakeTensor([0,3], tensDat3);

OUTPUT(tens7, NAMED('tens7'));
retDat7 := Tensor.R4.getData(tens7);
OUTPUT(retDat7, NAMED('retDat7'));

// Add the tensor to itself
tSum := Tensor.R4.Add(tens7, tens7);
OUTPUT(tSum, NAMED('SumDense'));

tSum2 := Tensor.R4.Add(tens3, tens3);
OUTPUT(tSum2, NAMED('SumSparse'));
