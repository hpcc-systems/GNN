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

// Tests the Reshape function

IMPORT GNN;
IMPORT GNN.Tensor;
IMPORT Tensor.R4 AS R4;

TensData := R4.TensData;

TensorDS := DATASET([{[1,1,1], 1.1},
                    {[1,2,1], 1.2},
                    {[1,3,1], 1.3}], TensData);

TestTensor := R4.MakeTensor([0,3, 2], TensorDS);

// The product of everything after the first index does not match
// that of the test tensor's product, so it should return an empty dataset
Result1 := R4.Reshape(TestTensor, [0, 1, 3]);
OUTPUT(IF(NOT Exists(Result1), 'Pass', 'Fail'), NAMED('Test1'));

// The products of everything after the first index do match,
// so the resulting shape should be equal to what was passed in
Result2 := R4.Reshape(TestTensor, [0, 2, 3]);
Expected := [0, 2, 3];
OUTPUT(Result2, {passing := IF(shape = Expected, 'Pass', 'Fail')}, NAMED('Test2'));
