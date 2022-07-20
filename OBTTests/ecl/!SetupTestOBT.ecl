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

//nokey

// Setup test from the tests folder. Confirms if the reason why
// a test fails has to do with the setup when tested in the
// OBT system

IMPORT Python3 AS Python;
IMPORT Std.System.Thorlib;

nNodes := Thorlib.nodes();
nodeId := Thorlib.node();

testRec := RECORD
  UNSIGNED node;
  STRING tf_version;
  STRING result := '';
END;

DATASET(testRec) DoTest := FUNCTION
  DATASET(testRec) testTF(DATASET(testRec) test) := EMBED(Python: activity)
    import traceback as tb
    nodeId = 999
    v = 0
    try:
      for tr in test:
        # Should only be one.
        nodeId, unused, result = tr
        try:
          import tensorflow.compat.v1 as tf # V2.x
          tf.disable_v2_behavior()
        except:
          try:
            import tensorflow as tf # V 1.x
          except:
            return [(nodeId, 'TensorFlow could not be found. Make sure it is installed.', tb.format_exc())]
        s = tf.Session()
        v = tf.__version__
        return [(nodeId, v, 'SUCCESS')]
    except:
      return [(nodeId, 'TensorFlow could not be found. Make sure it is installed.', tb.format_exc())]
  ENDEMBED;
  inp := DATASET([{nodeId, '', ''}], testRec, LOCAL);
  outp := testTF(inp);
  RETURN ASSERT(outp, result = 'SUCCESS', 'Error on node ' + node + ': ' + result);
END;

OUTPUT(DoTest, NAMED('Result'));