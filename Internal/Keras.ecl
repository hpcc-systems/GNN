/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT PYTHON3 AS PYTHON;
IMPORT $.^ AS GNN;
IMPORT GNN.Types;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Tensor;
IMPORT Std.System.Thorlib;

initParms := iTypes.initParms;
t_Tensor := Tensor.R4.t_Tensor;
kString := iTypes.kString;
losses := iTypes.losses;
metrics := Types.metrics;
FuncLayerDef := Types.FuncLayerDef;
nNodes := Thorlib.nodes();
node := Thorlib.node();
/**
  * This module provides an interface into Keras using Python embedding.
  *
  * <p>It uses global Python memory and a named scope to perform Keras functions.
  *
  * <p>The Init(...) function initializes the global environment and defines
  * a frequently used set of functions so that they don't need to be replicated
  * and created on each individual embed.
  * <p>Note that most functions receive a "seqId" parameter that is ignored by
  * the function.  This is used to control the order of execution of ECL statements
  * that call these functions.
  */
EXPORT Keras := MODULE
  SHARED globalScope := 'keras_' + node + '.ecl';
  /**
    * Initialize the Keras / Tensorflow environment and the global space.
    * Create a set of global functions for common operations.  This is to
    * improve performance and reduce replicaton of code between embeds.
    * Needed parameters are conveyed via the initParms dataset (initdata).
    * Note the activity flag on the embed.  The activity flag plus the
    * use of STREAMED DATASETS in and out, ensure that this function will
    * be executed on every Thor slave node.
    */
  EXPORT STREAMED DATASET(kString) Init(STREAMED DATASET(initParms) initdata, UNSIGNED gpusperserver = 0) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    # Function to initialize all the global variables and functions.  This should
    # only be called once.
    def initGlobals():
      try:
        import tensorflow.compat.v1 as tf # V2.x
        tf.disable_v2_behavior()
      except:
        import tensorflow as tf # V 1.x
      from tensorflow.keras import layers
      import numpy as np
      import math
      global nodeId, nNodes, maxSliceLen
      # Initialize global variables
      #   Extract the initialization parameters from initdata
      for rec in initdata:
        nodeId, nNodes, maxSliceLen = rec # Should only be one record
      #   Model cache indexed by model id.
      global modcache
      modcache = {}
      #   Session cache indexed by model id.
      global sesscache
      sesscache = {}
      #   The next model id to allocate
      global nextModId
      nextModId = 0
      #   The following 3 variables are for record keeping on each model.
      #   They are stored as a dictionary keyed by the model id.
      global currEpoch
      currEpoch = {}
      global batchCount
      batchCount = {}
      global cumLoss
      cumLoss = {}
      global kStrTypeDict
      #   kStrTypeDict needs to be kept in sync with Internal/Types.kStrType
      #   The kString type is used for several different purposes, and the type
      #   field indicates the meaning of a given string.
      kStrTypeDict = {'layer':1, 'compile':2, 'json':3, 'status':4}
      global dTypeDict, dTypeDictR, DTypeSizeDict
      #   dTypeDict is used to convey the data type of a tensor.  It must be
      #   kept in sync with the Tensor data types in Tensor.ecl
      dTypeDict = {1:np.float32, 2:np.float64, 3:np.int32, 4:np.int64}
      dTypeDictR = {'float32':1, 'float64':2, 'int32':3, 'int64':4}
      #   Store the element size for each tensor data type.
      dTypeSizeDict = {1:4, 2:8, 3:4, 4:8}
      # Define some common functions
      global format_exc
      # format_exc is used to format an exception as a string so that we
      # can return it.  It indicates where and why an error occurred.
      def _format_exc(func=''):
        import traceback as tb
        exc = tb.format_exc(limit=2)
        if len(exc) < 100000:
          return func + ': ' + exc
        else:
          return func + ': ' + exc[:200] + ' ... ' + exc[-200:]
      format_exc = _format_exc

      # Assign GPUs to Thor nodes if needed.
      import os
      #this "CUDA VISIBLE DEVICES" will set which GPU a given Thor node will have access to
      #without this, each single Thor node will try and allocate memory on all GPUs, which will make it crash
      if gpusperserver > 0:
        numServers = nNodes / gpusperserver
        os.environ["CUDA_VISIBLE_DEVICES"]=str(math.floor(int(nodeId)/numServers))
      else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
      tf.reset_default_graph()

      # Convert an ECL Tensor dataset into a single numpy ndarray.
      global Tens2Np
      def _Tens2Np(tens, recordOriented = False):
        def addData(a, dat, pos, is_fixed_size):
          if is_fixed_size:
            a[pos:pos+len(dat)] = dat
          else:
            a = np.append(a, dat)
          return a
        def verifyShape(shape):
          if recordOriented and shape[0] != 0:
            raise Exception('Record Oriented tensors ' + \
              'as data input to Fit or Predict must have a zero first shape component. Shape = ' + str(shape) + '.')
          #assert not recordOriented or (recordOriented and shape[0] == 0), 'Keras.ecl: Tens2Np: Record Oriented tensors ' + \
          #  'as data input to Fit or Predict must have a zero first shape component. Shape = ' + str(shape) + '.'
        try:
          a = None
          tshape = []
          lastSlice = 0
          fullSize = 0
          # If the first shape component is non-zero, then this is a fixed size Tensor
          # and exact positions are important.  If not fixed sized, then we take the
          # records sequentially and don't fill gaps.  We determine size by the actual
          # records received.
          isFixedSize = False
          for rec in tens:
            node, wi, sliceId, shape, dataType, maxSliceSize, sliceSize, densedat, sparsedat = rec
            verifyShape(shape)
            dtype = dTypeDict[dataType]
            tshape = shape
            if a is None:
              fullSize = np.prod(shape)
              isFixedSize = fullSize != 0
              a = np.zeros((fullSize,), dtype)
            if not isFixedSize:
              sliceId = lastSlice + 1
            if isFixedSize and (sliceId - lastSlice  > 1):
              # Skipped all zeros slice(s).  Move up the lastSlice counter.
              lastSlice = sliceId - 1
            if densedat:
              # Dense decoding
              a = addData(a, dtype(densedat), lastSlice * maxSliceSize, isFixedSize)
            else:
              # Sparse decoding
              dat = np.zeros((sliceSize,), dtype)
              for offset, val in sparsedat:
                dat[offset] = dtype(val)
              a = addData(a, dat, lastSlice * maxSliceSize, isFixedSize)
            lastSlice = sliceId
          if tshape[0] == 0:
            recSize = np.prod(tshape[1:])
            recCount = a.size / recSize
            tshape[0] = int(round(recCount))
          a = np.reshape(a, tshape)
          return a
        except:
          assert 1 == 0, format_exc('Tens2Np')
          return None
      Tens2Np = _Tens2Np

      # Convert a numpy ndarray into a Tensor dataset. Yield is used to
      # return one dataset record at a time.
      # maxSliceOverride allows a setting of the max slice size for this
      # tensor.
      # isWeights = True causes tensor to be broken into at least one
      # slice per HPCC cluster node, improving the efficiency of weight
      # synchronization.
      global Np2Tens
      # Returns a streamed dataset of t_Tensor
      def _Np2Tens(a, wi=0, maxSliceOverride=0, isWeights = False):
        try:
          epsilon = .000000001
          origShape = list(a.shape)
          flatA = a.reshape(-1)
          flatSize = flatA.shape[0]
          currSlice = 1
          indx = 0
          datType = dTypeDictR[str(a.dtype)]
          elemSize = dTypeSizeDict[datType]
          if maxSliceOverride:
            maxSliceSize = maxSliceOverride
          else:
            maxSliceSize = divmod(maxSliceLen, elemSize)[0]
          if isWeights and nNodes > 1 and flatSize > nNodes:
            # When we are synchronizing weights, we need to make sure
            # that we create Tensor with at least 1 slice per node.
            # This allows all nodes to participate equally in the
            # aggregation of weight changes.  For other data, it
            # is more efficient to return fewer slices.
            altSliceSize = math.ceil(flatSize / nNodes)
            maxSliceSize = min([maxSliceSize, altSliceSize])
          while indx < flatSize:
            remaining = flatSize - indx
            if remaining >= maxSliceSize:
              sliceSize = maxSliceSize
            else:
              sliceSize = remaining
            dat = list(flatA[indx:indx + sliceSize])
            dat = [float(d) for d in dat]
            elemCount = 0
            for i in range(len(dat)):
              if abs(dat[i]) > epsilon:
                elemCount += 1
            if elemCount > 0 or currSlice == 1:
              if elemCount * (elemSize + 4) < len(dat):
                # Sparse encoding
                sparse = []
                for i in range(len(dat)):
                  if abs(dat[i]) > epsilon:
                    sparse.append((i, dat[i]))
                yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, [], sparse)
              else:
                # Dense encoding
                yield (nodeId, wi, currSlice, origShape, datType, maxSliceSize, sliceSize, dat, [])
            currSlice += 1
            indx += sliceSize
        except:
          assert 1 == 0, format_exc('NP2Tens')
      Np2Tens = _Np2Tens
      global NpList2Tens

      # Convert a list of numpy ndarrays into an ECL tensor dataset.  Uses wi's to
      # distinguish the multiple tensors in the same dataset.
      # If isWeights is True, then we will create at least one tensor slice per node
      # for each tensor.  This allows more efficient merging of the weights.
      def _NpList2Tens(alist, isWeights = False):
        for i in range(len(alist)):
          for rec in Np2Tens(alist[i], i+1, isWeights = isWeights):
            yield rec
      NpList2Tens = _NpList2Tens
      global Tens2NpList

      # Convert an ECL tensor list dataset into a list of numpy ndarrays.
      # The wi field is used to distinguish the tensors in the list.
      def _Tens2NpList(tens, recordOriented = False):
        npList = []
        slices = []
        currWi = 1
        for slice in tens:
          node = slice[0]
          wi = slice[1]
          if wi != currWi:
            npList.append(Tens2Np(slices, recordOriented=recordOriented))
            currWi = wi
            slices = []
          slices.append(slice)
        if slices:
          npList.append(Tens2Np(slices, recordOriented=recordOriented))
        return npList
      Tens2NpList = _Tens2NpList
      # END OF InitGlobals
    # Only define the globals once, no matter how many times Init gets called.
    # Use the model cache (modcache) as a flag to determine if we've already
    # initialized.
    import threading
    global threadlock
    # Make sure we only intialize once.  Avoid reentrancy if called on multiple threads.
    threadlock = threading.Lock()
    threadlock.acquire()
    try:
      mc = modcache
    except:
      # modcache doesn't exist.  Do the initialization.
      try:
        # Initialize globals and define commonly used functions so that
        # they don't need to be repeated for each embed.
        # Global references to each function are stored in the global namespace.
        initGlobals()
      except:
        # We had an exception.  Format and return it.
        return [(nodeId, 1,4, format_exc('Init'))]
    finally:
      # Always release the threadlock, success or fail.
      threadlock.release()
    # Success.  Return blank status.
    return [(nodeId, 1, kStrTypeDict['status'], '')]
  ENDEMBED; // Init()

  /** Function to Define the model layers and (optionally) compile the model.
    * Returns a kString dataset.  An empty string indicates success.  Otherwise
    * the kString record contains an error message.
    * DefineModel gets called on each node of the cluster.
    */
  EXPORT STREAMED DATASET(kString) DefineModel(STREAMED DATASET(kString) mdef, UNSIGNED4 seqId)
                      := EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    from tensorflow.keras import layers
    global nextModId
    try:
      # Allocate a new modelId
      # Make sure we do it atomically to avoid conflict with
      # another model running on another thread
      threadlock.acquire()
      modId = nextModId
      nextModId += 1
      threadlock.release()
      # Create a new keras / tensorflow context.  It sometimes gets lost between calls,
      # so we explicitly restore it before each call that uses it.
      # Note that for each model, we create a new session and new graph under the hood.
      # The graph is stored within the session, so only the session and model are stored,
      # both by model id.
      graph = tf.Graph()
      with graph.as_default():
        tfSession = tf.Session()
        with tfSession.as_default():
          mod = tf.keras.Sequential()
          for rec in mdef:
            if rec[0] != nodeId:
              # Make sure we are only processing data meant for this node.
              continue
            rectype = rec[2]
            # If it is a layer definition string.  Add it to the model.
            if rectype == kStrTypeDict['layer']:
              mod.add(eval(rec[3]))
            # If it's a compile string, use it to compile the model.  All
            # layer strings need to precede any compile strings.  Only one
            # compile string should be supplied.
            elif rectype == kStrTypeDict['compile']:
              exec('mod.' + rec[3])
          # For some reason we need to do a get_weights / set_weights here, or set_weights
          # fails later???
          w = mod.get_weights()
          mod.set_weights(w)
          # Add this model to the model cache
          modcache[modId] = mod
          # And the session to the session cache
          sesscache[modId] = tfSession
      # We succeeded.  Return a blank status to indicate success.
      return [(nodeId, modId, kStrTypeDict['status'], '')]
    except:
      # We had an error.  Format the exception and return it in the kString
      return [(nodeId, 1, kStrTypeDict['status'], format_exc('DefineMod'))]
  ENDEMBED; // DefineModel
  /** Function to load a pre-trained Keras model and (optionally) compile the model.
    * Returns a kString dataset.  An empty string indicates success.  Otherwise
    * the kString record contains an error message.
    * DefineKAModel gets called on each node of the cluster.
    */
  EXPORT STREAMED DATASET(kString) DefineKAModel(STRING fname, STREAMED DATASET(kString) mdef, UNSIGNED4 seqId)
                      := EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    from tensorflow.keras import layers
    global nextModId
    try:
      # Allocate a new modelId
      # Make sure we do it atomically to avoid conflict with
      # another model running on another thread
      threadlock.acquire()
      modId = nextModId
      nextModId += 1
      threadlock.release()
      # Create a new keras / tensorflow context.  It sometimes gets lost between calls,
      # so we explicitly restore it before each call that uses it.
      # Note that for each model, we create a new session and new graph under the hood.
      # The graph is stored within the session, so only the session and model are stored,
      # both by model id.
      graph = tf.Graph()
      with graph.as_default():
        tfSession = tf.Session()
        with tfSession.as_default():
          for rec in mdef:
            if rec[0] != nodeId:
              # Make sure we are only processing data meant for this node.
              continue
            rectype = rec[2]
            # If it is a layer definition string.  Add it to the model.
            if rectype == kStrTypeDict['layer']:
              mod = eval ("tf.keras.applications." + fname + "(" + rec[3] + ")")
            # If it's a compile string, use it to compile the model.  All
            # layer strings need to precede any compile strings.  Only one
            # compile string should be supplied.
            elif rectype == kStrTypeDict['compile']:
              exec('mod.' + rec[3])
          # For some reason we need to do a get_weights / set_weights here, or set_weights
          # fails later???
          w = mod.get_weights()
          mod.set_weights(w)
          # Add this model to the model cache
          modcache[modId] = mod
          # And the session to the session cache
          sesscache[modId] = tfSession
      # We succeeded.  Return a blank status to indicate success.
      return [(nodeId, modId, kStrTypeDict['status'], '')]
    except:
      # We had an error.  Format the exception and return it in the kString
      return [(nodeId, 1, kStrTypeDict['status'], format_exc('DefineKAMod'))]
  ENDEMBED; // DefineKAModel
  /** Function to Define a Functional (i.e. Non-Sequential) model and (optionally)
    * compile the model.
    * Returns a kString dataset.  An empty string indicates success.  Otherwise
    * the kString record contains an error message.
    * DefineFuncModel gets called on each node of the cluster.
    */
  EXPORT STREAMED DATASET(kString) DefineFuncModel(STREAMED DATASET(FuncLayerDef) ldefs,
                                              UNSIGNED4 seqId,
                                              SET OF STRING inputs,
                                              SET OF STRING outputs,
                                              STRING cdef)
                      := EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    from tensorflow.keras import layers
    global nextModId
    try:
      # Allocate a new modelId
      # Make sure we do it atomically to avoid conflict with
      # another model running on another thread
      threadlock.acquire()
      modId = nextModId
      nextModId += 1
      threadlock.release()
      layerDict = {} # Temporary dictionary for keeping track of layers.
      predDict = {} # Temporary dict for keeping track of predecessors.
      # Create a new keras / tensorflow context.  It sometimes gets lost between calls,
      # so we explicitly restore it before each call that uses it.
      # Note that for each model, we create a new session and new graph under the hood.
      # The graph is stored within the session, so only the session and model are stored,
      # both by model id.
      graph = tf.Graph()
      with graph.as_default():
        tfSession = tf.Session()
        with tfSession.as_default():
          # Do two passes through the ldefs so that order of layers won't matter
          for rec in ldefs:
            lName, ldef, preds = rec
            newLayer = eval(ldef)
            layerDict[lName] = newLayer
            predDict[lName] = preds
          # Second pass to resolve the predecessors
          for name in layerDict.keys():
            layer = layerDict[name]
            predNames = predDict[name]
            lpreds = []
            for predName in predNames:
              pred = layerDict[predName]
              lpreds.append(pred)
            # Call the layer object's call method with the list of predecessors
            # to set the preds for that layer.
            if lpreds:
              if len(lpreds) == 1:
                layer = layer(lpreds[0])
              else:
                layer = layer(lpreds)
              layerDict[name] = layer
          # Now create the model using inputs and outputs
          inps = []
          outps = []
          for inpName in inputs:
            l = layerDict[inpName]
            inps.append(l)
          for outName in outputs:
            l = layerDict[outName]
            outps.append(l)
          mod = tf.keras.models.Model(inputs=inps, outputs=outps)
          # If there's a compile string, use it to compile the model.
          if cdef:
            exec('mod.' + cdef)
          # For some reason we need to do a get_weights / set_weights here, or set_weights
          # fails later???
          w = mod.get_weights()
          mod.set_weights(w)
          # Add this model to the model cache
          modcache[modId] = mod
          # And the session to the session cache
          sesscache[modId] = tfSession
      # We succeeded.  Return a blank status to indicate success.
      return [(nodeId, modId, kStrTypeDict['status'], '')]
    except:
      # We had an error.  Format the exception and return it in the kString.
      return [(nodeId, 1, kStrTypeDict['status'], format_exc('DefineFuncMod'))]
  ENDEMBED; // DefineFuncModel
  /**
    * Return a JSON string representing the layers of the model.  Does not return any
    * compile information or trained weights.
    */
  EXPORT STREAMED DATASET(kString) ToJSON(STREAMED DATASET(kString) dummy, UNSIGNED4 seqId,
                  UNSIGNED modelid = 0) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      # Restore the keras / tensorflow context for this model.
      mod = modcache[modelid]
      tfSession = sesscache[modelid]
      with tfSession.as_default():
        with tfSession.graph.as_default():
          js = mod.to_json()
      # Succeeded.  Return a blank status.
      return [(nodeId, 1, kStrTypeDict['status'], js)]
    except:
      # Failed.  Forat an exception and send it.
      return [(nodeId, 1, 4, format_exc('ToJSON'))]
  ENDEMBED;
  /**
    * Construct a Keras model from the JSON string passed in.
    */
  EXPORT STREAMED DATASET(kString) FromJSON(STREAMED DATASET(kString) ksjson, UNSIGNED4 seqId)
              := EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    global nextModId
    # Should be only one record on each node
    try:
      json = 'EMPTY'
      for rec in ksjson:
        # Should only be one json kString record.
        json = rec[2]
      # Restore the keras / tensorflow context for this model.
      graph = tf.Graph()
      with graph.as_default():
        tfSession = tf.Session()
        with tfSession.as_default():
           mod = tf.keras.models.model_from_json(json)
      modId = nextModId
      nextModId += 1
      modcache[modId] = mod
      sesscache[modId] = tfSession
    except:
      # Error.  Return an exception string.
      return [(nodeId, 1, kStrTypeDict['status'], format_exc('FromJSON'))]
    # Success. Return an empty string.
    return [(nodeId, modId, kStrTypeDict['status'], '')]
  ENDEMBED;
  /**
    * Compile a previously defined model.
    */
  EXPORT STREAMED DATASET(kString) CompileMod(STREAMED DATASET(kString) compilestr, UNSIGNED4 seqId,
                UNSIGNED modelid = 0) := EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    # Restore the keras / tensorflow context for this model.
    tfSession = sesscache[modelid]
    mod = modcache[modelid]
    with tfSession.as_default():
      with tfSession.graph.as_default():
        # Should only have one compilestr record per node
        try:
          cstr = 'EMPTY'
          for rec in compilestr:
            cstr = rec[2]
          exec('mod.' + cstr)
        except:
          return [(nodeId, 1, kStrTypeDict['status'], format_exc('CompileMod'))]
        return [(nodeId, 1, kStrTypeDict['status'], '')]
  ENDEMBED;
  /**
    * Get the weights from the Keras / Tensorflow model.
    * Weights are returned as a Tensor List.
    */
  EXPORT STREAMED DATASET(t_Tensor) GetWeights(
                          STREAMED DATASET(kString) dummy, UNSIGNED4 seqId, UNSIGNED modelid = 0) :=
                            EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    threadlock.acquire()
    try:
      # Restore the keras / tensorflow context for this model.
      tfSession = sesscache[modelid]
      mod = modcache[modelid]
      with tfSession.as_default():
        with tfSession.graph.as_default():
          w = mod.get_weights()
      return NpList2Tens(w, isWeights = True)
    except:
      # IF there was an error, return an empty dataset.
      assert 1 == 0, format_exc('GetWeights modelId = ' + str(modelid))
      return []
    finally:
      threadlock.release()
  ENDEMBED;
  /**
    * Set the weights into the Keras / TF model.  The weights are sent as
    * a Tensor List (Tensor dataset), one Tensor per layer.
    */
  EXPORT STREAMED DATASET(kString) SetWeights(STREAMED DATASET(t_Tensor) tens, UNSIGNED4 seqId,
              UNSIGNED modelid = 0) := EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    # Restore the keras / tensorflow context for this model.
    tfSession = sesscache[modelid]
    mod = modcache[modelid]
    try:
      # Restore the Keras / TF context.
      tfSession = sesscache[modelid]
      mod = modcache[modelid]
      w = Tens2NpList(tens)
      with tfSession.as_default():
        with tfSession.graph.as_default():
          mod.set_weights(w)
      # Success.  Return an empty status string.
      return [(nodeId, 1, 1, '')]
    except:
      # An error occurred.  Return a formatted exception string.
      return [(nodeId, 1,1, format_exc('SetWeights'))]
  ENDEMBED;
  /**
    * Process a single batch of training data.  The starting weights
    * are sent as an input parameter (Tensor List).  Weight changes
    * as a result of the training are returned (Tensor List).
    * x and y represent the independent and dependent training data.
    */
  EXPORT STREAMED DATASET(t_Tensor) FitBatch(
              STREAMED DATASET(t_Tensor) weights,
              STREAMED DATASET(t_Tensor) x,
              STREAMED DATASET(t_Tensor) y,
              UNSIGNED4 seqId,
              UNSIGNED4 epoch,
              UNSIGNED modelid = 0,
              UNSIGNED4 kbatchsize = 32,
              REAL lr = 1.0) :=
            EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      import tensorflow.compat.v1 as tf # V2.x
      tf.disable_v2_behavior()
    except:
      import tensorflow as tf # V 1.x
    import numpy as np
    global currEpoch, batchCount, cumLoss
    try:
      # Accumulate the loss for each epoch.
      if epoch != currEpoch.get(modelid, 0):
        batchCount[modelid] = 0
        cumLoss[modelid] = 0.0
        currEpoch[modelid] = epoch
      # Process this batch.
      batchCount[modelid] += 1
      wA_changes = []
      # Restore Keras / TF context
      mod = modcache[modelid]
      # Convert the incoming weights to a list of numpy arrays
      wA = Tens2NpList(weights)
      # Convert the X tensor to a numpy array
      xAL = Tens2NpList(x, recordOriented = True)
      # Convert the Y tensor to a numpy array
      yAL = Tens2NpList(y, recordOriented = True)
      if xAL and yAL and xAL[0].size > 0 and yAL[0].size > 0:
        # We've got some data
        # Do some error checking.
        for i in range(len(xAL)):
          xA = xAL[i]
          yA = yAL[i]
          if xA.size == 0 or yA.size == 0 or xA.shape[0] != yA.shape[0]:
            assert 1 == 0, 'Fit: X and Y sizes do not match or are zero: xShape = ' + str(xA.shape) + ', yShape = ' + str(yA.shape)
        # Restore the keras / tensorflow context for this model.
        tfSession = sesscache[modelid]
        with tfSession.as_default():
          with tfSession.graph.as_default():
            # Set the starting weights
            mod.set_weights(wA)
            # Run one batch to fit the model
            tfHistory = mod.fit(xAL, yAL, epochs=epoch, batch_size=kbatchsize, initial_epoch=epoch-1, shuffle=False, steps_per_epoch = 1)
            # Update the cumulative (epoch) loss
            currLoss = tfHistory.history['loss'][-1]
            cumLoss[modelid] += currLoss
            # Get the new weights from Keras model.
            wA_out = mod.get_weights()
        # For each layer, subtract the new weights from the starting weights to compute
        # the weight updates.  Scale the changes by the learningRate (lr) so that we can
        # control the lr as a fraction of the learing rate used within the optimizer from compileMod.
        for i in range(len(wA)):
          wA_changes.append((wA_out[i] - wA[i]) * lr)
      else:
        # No X / Y data received.  Send null changes
        for i in range(len(wA)):
          wA_changes.append(np.zeros_like(wA[i]))
      # Return the weight changes as a Tensor List.
      return NpList2Tens(wA_changes, isWeights = True)
    except:
      # Error occurred, but no string returned.  So we do an assert to convey the error.
      assert 1 == 0, format_exc('FitBatch')
  ENDEMBED; // FitBatch
  /**
    * Get the current epoch's accumulated average loss up to this point.
    */
  EXPORT STREAMED DATASET(losses) GetLoss(STREAMED DATASET(kString) dummy, UNSIGNED4 seqId,
        UNSIGNED modelid = 0):=
        EMBED(Python: globalscope(globalScope), persist('query'), activity)
    global batchCount, cumLoss
    try:
      assert batchCount[modelid] > 0, 'Keras.GetLoss: batchCount = 0' + ', currEpoch = ' + str(currEpoch[modelid])
      loss = cumLoss[modelid] / batchCount[modelid]
    except:
      assert False, format_exc('GetLoss -- modelId = ' + str(modelid) + ', batchCount = ' + str(batchCount))
      return [(0.0,)]
    return [(loss,)]
  ENDEMBED;
  /**
    * Evaluate a model against test data (x) and expected result (y).
    * A set of metrics are returned.  The first metric is the loss,
    * followed by any other metrics defined within the model compile string.
    */
  EXPORT STREAMED DATASET(metrics) Evaluate(
              STREAMED DATASET(t_Tensor) x,
              STREAMED DATASET(t_Tensor) y,
              UNSIGNED4 seqId,
              UNSIGNED modelid = 0) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      mod = modcache[modelid]
      # Convert x data to a numpy array
      xA = Tens2NpList(x, recordOriented = True)
      # Convert y data to a numpy array
      yA = Tens2NpList(y, recordOriented = True)
      outRecs = []
      # Restore the keras / tensorflow context for this model.
      tfSession = sesscache[modelid]
      with tfSession.as_default():
        with tfSession.graph.as_default():
          # Evaluate the Keras model
          metrics = mod.evaluate(xA, yA)
          # Get the name for each metric
          mNames = mod.metrics_names
          for i in range(len(metrics)):
            # Return the name and value for each defined metric.
            rec = (i, mNames[i], float(metrics[i]))
            outRecs.append(rec)
      return outRecs
    except:
      # Error occurred, but no string returned.  So we do an assert to convey the error.
      assert 1 == 0, format_exc('Evaluate')
  ENDEMBED;
  /**
    * Use the Keras model to predict the output for a set
    * of independent (x) data.
    */
  EXPORT DATASET(t_Tensor) Predict(
            DATASET(t_Tensor) xDat,
            UNSIGNED4 seqId,
            UNSIGNED modelId) := FUNCTION
      STREAMED DATASET(t_Tensor) Predict2(
                STREAMED DATASET(t_Tensor) x_dat,
                UNSIGNED4 seq_id,
                UNSIGNED model_id) :=
                      EMBED(Python: globalscope(globalScope), persist('query'), activity)
        import numpy as np
        # Generator function for producing the predictions
        def predGen(mod, tfSession):
          try:
            # We need to process the data one slice at a time, so that we can emit
            # slices with the proper wi and sliceId so that the record indexes line up
            # between the supplied x and the returned predictions.
            xAL = [] # X array list accumulates wi's for each sliceId
            currSlice = 0
            maxRecs = 0  # Should be the same for each wi across all slices
            # process all the wi's for each sliceId.  Inputs should be aligned at this point.
            # and sorted by sliceId, wi.
            for slice in x_dat:
              node, wi, sliceId, shape, dataType, maxSliceSize, slice_size, \
                        densedat, sparsedat = slice
              # Calculate the maximum number of records per slice.  This only needs to be
              #  done for the first slice and wi, since it should be consistent for aligned
              #  tensors
              if maxRecs == 0:
                recSize = np.prod(shape[1:])
                maxRecs = int(maxSliceSize / recSize)
              if sliceId != currSlice:
                # Got the next slice.  Now we should have all the wi's for the previous slice.
                # Process the full slice.
                if xAL:
                  # We have a slice accumulated.
                  # Process it.
                  # Restore the keras / tensorflow context for this model.
                  with tfSession.as_default():
                    with tfSession.graph.as_default():
                      predA = mod.predict(xAL, steps=1)
                  for i in range(len(predA)):
                    sliceA = predA[i]
                    recSize = int(np.prod(sliceA.shape[1:]))
                    newMaxSize = maxRecs * recSize
                    # Np2Tens will set the sliceId to 1 since it's the only slice.
                    # so we need to set the sliceId back to the original
                    for s in Np2Tens(sliceA, wi=i+1, maxSliceOverride=newMaxSize):
                      # Should only be one slice since we forced the maxSliceOverride
                      # but it's a generator, so we need to do for loop.
                      s = s[:2] + (currSlice,) + s[3:]
                      yield s
                  # Now clear the accumultor
                  xAL = []
                currSlice = sliceId
              # Convert the slice to a numpy array and add it to the accumulator
              # Force the sliceId to 1 to handle this slice as standalone
              sliceAdj = [node, wi, 1, shape, dataType, maxSliceSize, slice_size, densedat, sparsedat]
              xA = Tens2Np([sliceAdj], recordOriented = True)
              xAL.append(xA)
              # END for slice in x_dat
            # Process the last sliceId
            if xAL:
              # Restore keras / tf context
              with tfSession.as_default():
                with tfSession.graph.as_default():
                  predA = mod.predict(xAL, steps=1)
              if type(predA) != type([]):
                predA = [predA]
              for i in range(len(predA)):
                sliceA = predA[i]
                recSize = int(np.prod(sliceA.shape[1:]))
                newMaxSize = maxRecs * recSize
                # Np2Tens will set the sliceId to 1 since it's the only slice.
                # so we need to set the sliceId back to the original
                for s in Np2Tens(sliceA, wi=i+1, maxSliceOverride=newMaxSize):
                  # Should only be one slice since we forced the maxSliceOverride
                  # but it's a generator, so we need to do for loop.
                  s = s[:2] + (currSlice,) + s[3:]
                  yield s
            return
          except:
            # An error occured during predGen()
            assert 0 == 1, format_exc('Predict2 -- predGen')
        # END predGen()
        try:
          # Get the model
          mod = modcache[model_id]
          sess = sesscache[model_id]
          # Return the generator that will produce the output Tensor list
          return predGen(mod, sess)
        except:
          # An error occurred during Predict.
          assert 0 == 1, format_exc('Predict2')
          return []
      ENDEMBED; // Predict2
    // Sort Xdat by sliceId and then by wi so that we can present the model with all
    // inputs (i.e. wi's) at once.
    xDatS := SORT(xDat, sliceId, wi, LOCAL);
    preds := predict2(xDatS, seqId, modelId);
    // Now re-sort into the cannonical order
    predsS := SORT(preds, wi, sliceId, LOCAL);
    RETURN predsS;
  END; // Predict
  /**
    * Shutdown the Keras Interface and free up all global memory fields.
    * This leaves behind, at most, a small memory footprint that should
    * be reused for any subsequent calls.
    * This is not really required since subsequent calls will use the same
    * global memory, but is here for future use (e.g. when we support
    * multiple Keras models).  If needed, care should be taken to ensure
    * that Shutdown is actually executed (i.e. by outputting the results).
    */
  EXPORT STREAMED DATASET(kString) Shutdown(
              STREAMED DATASET(kString) temp,
              UNSIGNED4 seqId) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
      global nodeId, nNodes, maxSliceLen
      global modcache
      global sesscache
      global currEpoch
      global batchCount
      global cumLoss
      global kStrTypeDict
      global dTypeDict, dTypeDictR, DTypeSizeDict
      global Tens2Np, Np2Tens, Tens2NpList, NpList2Tens
      global format_exc
      nid = nodeId; # Hold onto nodeId for the return
      try:
        nodeId = None
        nNodes = None
        maxSliceLen = None
        currEpoch = None
        batchCount = None
        cumLoss = None
        kStrTypeDict = None
        dTypeDict = None
        dTypeDictR = None
        DTypeSizeDict = None
        Tens2Np = None
        Np2Tens = None
        Tens2NpList = None
        NpList2Tens = None
        format_exc = None
        del(modcache)
        del(sesscache)
        del(nodeId)
        del(nNodes)
        del(maxSliceLen)
        del(currEpoch)
        del(batchCount)
        del(cumLoss)
        del(kStrTypeDict)
        del(dTypeDict)
        del(DTypeSizeDict)
        del(Tens2Np)
        del(Np2Tens)
        del(Tens2NpList)
        del(NpList2Tens)
        del(format_exc)
        return [(nid, 1, 4, '')]
      except:
        return [(nid, 1, 4, format_exc())]
  ENDEMBED; // Shutdown
END; // Keras Module
