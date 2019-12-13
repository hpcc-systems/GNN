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
  STRING result := '';
END;
DATASET(testRec) DoTest := FUNCTION
  DATASET(testRec) testTF(DATASET(testRec) test) := EMBED(Python: activity)
    import traceback as tb
    nodeId = 999
    try:
      for tr in test:
        # Should only be one.
        nodeId, result = tr
        import tensorflow as tf
        s = tf.Session()
        return [(nodeId, 'SUCCESS')]
    except:
      return [(nodeId, tb.format_exc())]
  ENDEMBED;
  inp := DATASET([{nodeId, ''}], testRec, LOCAL);
  outp := testTF(inp);
  RETURN ASSERT(outp, result = 'SUCCESS', 'Error on node ' + node + ': ' + result);
END;

OUTPUT(DoTest, NAMED('Result'));