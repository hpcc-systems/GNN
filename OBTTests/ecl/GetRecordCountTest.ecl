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

// Tests the getRecordCount function

IMPORT GNN;
IMPORT GNN.Tensor;
IMPORT Tensor.R4 AS R4;

STRING IncorrectResult(INTEGER Expected, INTEGER Result) := FUNCTION
  RETURN 'Expected: ' + Expected + ' Result: ' + Result;
END;

TensData := R4.TensData;

TensorDS := DATASET([{[1,1], 1.1},
                    {[1,2], 1.2},
                    {[1,3], 1.3},
                    {[2,1], 2.1},
                    {[2,2], 2.2},
                    {[2,3], 2.3},
                    {[3,1], 3.1},
                    {[3,2], 3.2},
                    {[3,3], 3.3},
                    {[4, 1], 4.1},
                    {[4, 2], 4.2},
                    {[4, 3], 4.3},
                    {[5, 1], 5.1,},
                    {[5, 2], 5.2},
                    {[5, 3], 5.3}], TensData);

TestTensor := R4.MakeTensor([0,5], TensorDS);
Result1 := R4.GetRecordCount(TestTensor);
Expected1 := 5;

OUTPUT(IF(Result1 = Expected1, 'Correct', IncorrectResult(Expected1, Result1)), NAMED('Test1'));

TensorDS2 := DATASET([{[1,1], 1.1},
                    {[1,2], 1.2},
                    {[1,3], 1.3},
                    {[2,1], 2.1},
                    {[2,2], 2.2},
                    {[2,3], 2.3},
                    {[3,1], 3.1},
                    {[3,2], 3.2},
                    {[3,3], 3.3}], TensData);

TestTensor2 := R4.MakeTensor([0,3], TensorDS2);
Result2 := R4.GetRecordCount(TestTensor2);
Expected2 := 3;

OUTPUT(IF(Result2 = Expected2, 'Correct', IncorrectResult(Expected2, Result2)), NAMED('Test2'));
