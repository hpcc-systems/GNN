/*##############################################################################
    
    HPCC SYSTEMS software Copyright (C) 2022 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       
       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

IMPORT GNN;
IMPORT GNN.Tensor;
IMPORT Tensor.R4 AS R4;

// Test for the Add function

TensData := R4.TensData;

// Building first tensor
TensorDS1 := DATASET([{[1,1], 1.1},
                    {[1,2], 1.2},
                    {[1,3], 1.3},
                    {[2,1], 2.1},
                    {[2,2], 2.2},
                    {[2,3], 2.3},
                    {[3,1], 3.1},
                    {[3,2], 3.2},
                    {[3,3], 3.3}], TensData);

Tensor1 := R4.MakeTensor([0,3], TensorDS1);
T1Data := R4.getData(Tensor1);

// Building second tensor
TensorDS2 := DATASET([{[1,1], 5.1},
                    {[1,2], 5.2},
                    {[1,3], 5.3},
                    {[2,1], 6.1},
                    {[2,2], 6.2},
                    {[2,3], 6.3},
                    {[3,1], 7.1},
                    {[3,2], 7.2},
                    {[3,3], 7.3}], TensData);

Tensor2 := R4.MakeTensor([0, 3], TensorDS2);
T2Data := R4.getData(Tensor2);

// Add the two tensors
TensorAdd := R4.Add(Tensor1, Tensor2);
AddData := R4.getData(TensorAdd);

additionCheck := RECORD
    REAL Sum;
END;

// Sums the values for both tensors
additionCheck JoinThem(TensData L, TensData R) := TRANSFORM
    SELF.Sum := L.value + R.value;
END;

TensorSum := JOIN(T1Data, T2Data, LEFT.indexes = RIGHT.indexes, JoinThem(LEFT, RIGHT));

// Gets amount of rows that do not have the correct sum
// The values in TensorSum and AddData should match
NumNotMatching := COUNT(TensorSum) - COUNT(JOIN(TensorSum, AddData, LEFT.Sum = RIGHT.value));

// If > 0, the test failed
OUTPUT(IF(NumNotMatching = 0, 'Pass', 'Failure: ' + NumNotMatching + ' incorrect sums'), NAMED('Result'));
