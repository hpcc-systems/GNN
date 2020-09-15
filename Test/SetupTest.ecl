/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Validates that the Python3 and Tensorflow environments are set up correctly.
  * <p>Outputs a status for each Thor node.  The result for each node will
  * be 'SUCCESS' or a Python traceback if the test was unsuccessful on a given
  * node.
  */
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
          import tensorflow as tf # V 1.x
        s = tf.Session()
        v = tf.__version__
        return [(nodeId, v, 'SUCCESS')]
    except:
      return [(nodeId, 'unknown', tb.format_exc())]
  ENDEMBED;
  inp := DATASET([{nodeId, '', ''}], testRec, LOCAL);
  outp := testTF(inp);
  RETURN ASSERT(outp, result = 'SUCCESS', 'Error on node ' + node + ': ' + result);
END;

OUTPUT(DoTest, NAMED('Result'));