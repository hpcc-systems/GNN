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
  EXPORT STREAMED DATASET(kString) Init(STREAMED DATASET(initParms) initdata) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    # Function to initialize all the global variables and functions.  This should
    # only be called once.
    def initGlobals():
      import tensorflow as tf
      from tensorflow.keras import layers
      import numpy as np
      global nodeId, nNodes, maxSliceLen
      # Extract the initialization parameters from initdata
      for rec in initdata:
        nodeId, nNodes, maxSliceLen = rec # Should only be one record
      # Initialize global variables
      global modcache
      global tfHistory
      tfHistory = None
      global currEpoch
      currEpoch = 0
      global batchCount
      batchCount = 0
      global cumLoss
      cumLoss = 0
      global kStrTypeDict
      # kStrTypeDict needs to be kept in sync with Internal/Types.kStrType
      # The kString type is used for several different purposes, and the type
      # field indicates the meaning of a given string.
      kStrTypeDict = {'layer':1, 'compile':2, 'json':3, 'status':4}
      global dTypeDict, dTypeDictR, DTypeSizeDict
      # dTypeDict is used to convey the data type of a tensor.  It must be
      # kept in sync with the Tensor data types in Tensor.ecl
      dTypeDict = {1:np.float32, 2:np.float64, 3:np.int32, 4:np.int64}
      dTypeDictR = {'float32':1, 'float64':2, 'int32':3, 'int64':4}
      # Store the element size for each tensor data type.
      dTypeSizeDict = {1:4, 2:8, 3:4, 4:8}
      modcache = {}
      # Define some common functions
      global format_exc
      # format_exc is used to format an exception as a string so that we
      # can return it.  It indicates where and why an error occurred.
      def _format_exc(func=''):
        import traceback as tb
        exc = tb.format_exc()
        if len(exc) < 10000:
          return func + ': ' + exc
        else:
          return func + ': ' + exc[:200] + ' ... ' + exc[-200:]
      format_exc = _format_exc

      # Convert an ECL Tensor dataset into a single numpy ndarray.
      global Tens2Np
      def _Tens2Np(tens):
        def addData(a, dat, pos, is_fixed_size):
          if is_fixed_size:
            a[pos:pos+len(dat)] = dat
          else:
            a = np.append(a, dat)
          return a
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
      global Np2Tens
      # Returns a streamed dataset of t_Tensor
      def _Np2Tens(a, wi=0, maxSliceOverride=0):
        try:
          epsilon = .000000001
          origShape = list(a.shape)
          flatA = a.reshape(-1)
          flatSize = flatA.shape[0]
          sliceId = 1
          indx = 0
          datType = dTypeDictR[str(a.dtype)]
          elemSize = dTypeSizeDict[datType]
          if maxSliceOverride:
            maxSliceSize = maxSliceOverride
          else:
            maxSliceSize = divmod(maxSliceLen, elemSize)[0]
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
            if elemCount > 0 or sliceId == 1:
              if elemCount * (elemSize + 4) < len(dat):
                # Sparse encoding
                sparse = []
                for i in range(len(dat)):
                  if abs(dat[i]) > epsilon:
                    sparse.append((i, dat[i]))
                yield (nodeId, wi, sliceId, origShape, datType, maxSliceSize, sliceSize, [], sparse)
              else:
                # Dense encoding
                yield (nodeId, wi, sliceId, origShape, datType, maxSliceSize, sliceSize, dat, [])
            sliceId += 1
            indx += sliceSize
        except:
          assert 1 == 0, format_exc('NP2Tens')
      Np2Tens = _Np2Tens
      global NpList2Tens
      # Convert an ECL tensor list dataset into a list of numpy ndarrays.
      # The wi field is used to distinguish the tensors in the list.
      def _NpList2Tens(alist):
        for i in range(len(alist)):
          for rec in Np2Tens(alist[i], i+1):
            yield rec
      NpList2Tens = _NpList2Tens
      global Tens2NpList
      # Convert a list of numpy ndarrays into an ECL tensor dataset.  Uses wi's to
      # distinguish the multiple tensors in the same dataset.
      def _Tens2NpList(tens):
        npList = []
        slices = []
        currWi = 1
        for slice in tens:
          node = slice[0]
          wi = slice[1]
          if wi != currWi:
            npList.append(Tens2Np(slices))
            currWi = wi
            slices = []
          slices.append(slice)
        if slices:
          npList.append(Tens2Np(slices))
        return npList
      Tens2NpList = _Tens2NpList
      # END OF InitGlobals
    # Only define the globals once, no matter how many times Init gets called.
    # Use the model cache (modcache) as a flag to determine if we've already
    # initialized.
    try:
      mc = modcache
    except:
      # modcache doesn't exist.  Do the initialization.
      try:
        initGlobals()
      except:
        # We had an exception.  Format and return it.
        return [(nodeId, 1,4,tb.format_exc('Init'))]
    # Success.  Return blank status.
    return [(nodeId, 1, kStrTypeDict['status'], '')]
  ENDEMBED; // Init()

  /** Function to Define the model layers and (optionally) compile the model.
    * Returns a kString dataset.  An empty string indicates success.  Otherwise
    * the kString record contains an error message.
    * DefineModel gets called on each node of the cluster.
    */
  EXPORT STREAMED DATASET(kString) DefineModel(STREAMED DATASET(kString) mdef, UNSIGNED4 sess) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import traceback as tb
    import tensorflow as tf
    from tensorflow.keras import layers
    global tfSession
    try:
      # Restore the keras / tensorflow context.  It sometimes gets lost between calls,
      # so we explicitly restore it before each call that uses it.
      tfSession = tf.keras.backend.get_session()
      with tfSession.as_default():
        with tfSession.graph.as_default():
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
            modcache['default'] = mod
          # For some reason we need to do a get_weights / set_weights here, or set_weights
          # fails later???
          w = mod.get_weights()
          mod.set_weights(w)
        # We succeeded.  Return a blank status to indicate success.
        return [(nodeId, 1, kStrTypeDict['status'], '')]
    except:
      # We had an error.  Format the exception and return it in the kString
      return [(nodeId, 1, kStrTypeDict['status'], format_exc('DefineMod'))]
  ENDEMBED;
  /**
    * Return a JSON string representing the layers of the model.  Does not return any
    * compile information or trained weights.
    */
  EXPORT STREAMED DATASET(kString) ToJSON(STREAMED DATASET(kString) dummy, UNSIGNED4 model) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    try:
      mod = modcache['default']
      # Succeeded.  Return a blank status.
      return [(nodeId, 1, kStrTypeDict['status'], mod.to_json())]
    except:
      # Failed.  Forat an exception and send it.
      return [(nodeId, 1, 4, format_exc('ToJSON'))]
  ENDEMBED;
  /**
    * Construct a Keras model from the JSON string passed in.
    */
  EXPORT STREAMED DATASET(kString) FromJSON(STREAMED DATASET(kString) ksjson, UNSIGNED4 session) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import tensorflow as tf
    # Should be only one record on each node
    try:
      json = 'EMPTY'
      for rec in ksjson:
        # Should only be one json kString record.
        json = rec[2]
      mod = tf.keras.models.model_from_json(json)
      modcache['default'] = mod
    except:
      # Error.  Return an exception string.
      return [(nodeId, 1, kStrTypeDict['status'], format_exc('FromJSON'))]
    # Success. Return an empty string.
    return [(nodeId, 1, kStrTypeDict['status'], '')]
  ENDEMBED;
  /**
    * Compile a previously defined model.
    */
  EXPORT STREAMED DATASET(kString) CompileMod(STREAMED DATASET(kString) compilestr, UNSIGNED4 model) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import tensorflow as tf
    tf.keras.backend.set_session(tfSession)
    mod = modcache['default']
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
                          STREAMED DATASET(kString) dummy, UNSIGNED4 model) :=
                            EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import tensorflow as tf
    # Restore the Keras / TF context.
    tf.keras.backend.set_session(tfSession)
    try:
      mod = modcache['default']
      w = mod.get_weights()
      return NpList2Tens(w)
    except:
      # IF there was an error, return an empty dataset.
      return []
  ENDEMBED;
  /**
    * Set the weights into the Keras / TF model.  The weights are sent as
    * a Tensor List (Tensor dataset), one Tensor per layer.
    */
  EXPORT STREAMED DATASET(kString) SetWeights(STREAMED DATASET(t_Tensor) tens, UNSIGNED4 model) :=
                        EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import tensorflow as tf
    import traceback as tb
    # Restore the Keras / TF context.
    tf.keras.backend.set_session(tfSession)
    try:
      w = Tens2NpList(tens)
      mod = modcache['default']
      #w2 = mod.get_weights()
      #mod.set_weights(w)
      outStr = ''
      # Success.  Return an empty status string.
      return [(nodeId, 1, 1, outStr)]
    except:
      # An error occurred.  Return a formatted exception string.
      return [(nodeId, 1,1,tb.format_exc('SetWeights')[-500:])]
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
              UNSIGNED4 model,
              UNSIGNED4 epoch) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import traceback as tb
    import tensorflow as tf
    import numpy as np
    global currEpoch, batchCount, cumLoss
    try:
      # Accumulate the loss for each epoch.
      if epoch != currEpoch:
        batchCount = 0
        cumLoss = 0.0
        currEpoch = epoch
      # Process this batch.
      batchCount += 1
      wA_changes = []
      # Restore Keras / TF context
      tf.keras.backend.set_session(tfSession)
      mod = modcache['default']
      # Convert the incoming weights to a list of numpy arrays
      wA = Tens2NpList(weights)
      # Convert the X tensor to a numpy array
      xAL = Tens2NpList(x)
      # Convert the Y tensor to a numpy array
      yAL = Tens2NpList(y)
      # Do some error checking.
      if xAL and yAL and xAL[0].size > 0 and yAL[0].size > 0:
        xA = xAL[0]
        yA = yAL[0]
        if xA.size == 0 or yA.size == 0 or xA.shape[0] != yA.shape[0]:
          assert 1 == 0, 'Fit: X and Y sizes do not match or are zero: xShape = ' + str(xA.shape) + ', yShape = ' + str(yA.shape)
        # More Keras TF context restoration
        with tfSession.as_default():
          with tfSession.graph.as_default():
            global tfHistory
            # Set the starting weights
            mod.set_weights(wA)
            # Run one batch to fit the model
            tfHistory = mod.fit(xA, yA, epochs=epoch, batch_size=32, initial_epoch=epoch-1, shuffle=False, steps_per_epoch = 1)
            # Update the cumulative (epoch) loss
            currLoss = tfHistory.history['loss'][-1]
            cumLoss += currLoss
            # Get the new weights from Keras model.
            wA_out = mod.get_weights()
        # For each layer, subtract the new weights from the starting weights to compute
        # the weight updates.
        for i in range(len(wA)):
          wA_changes.append(wA_out[i] - wA[i])
      else:
        # No X / Y data received.  Send null changes
        for i in range(len(wA)):
          wA_changes.append(np.zeros_like(wA[i]))
      # Return the weight changes as a Tensor List.
      return NpList2Tens(wA_changes)
    except:
      # Error occurred, but no string returned.  So we do an assert to convey the error.
      assert 1 == 0, format_exc('FitBatch')
  ENDEMBED;
  /**
    * Get the current epoch's accumulated average loss up to this point.
    */
  EXPORT STREAMED DATASET(losses) GetLoss(STREAMED DATASET(kString) dummy, UNSIGNED4 model):=
        EMBED(Python: globalscope(globalScope), persist('query'), activity)
    assert batchCount > 0, 'Keras.GetLoss: batchCount = 0' + ', currEpoch = ' + str(currEpoch)
    loss = cumLoss / batchCount
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
              UNSIGNED4 model) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    mod = modcache['default']
    # Convert x data to a numpy array
    xA = Tens2NpList(x)
    # Convert y data to a numpy array
    yA = Tens2NpList(y)
    outRecs = []
    # Restore Keras / TF context
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
  ENDEMBED;
  /**
    * Use the Keras model to predict the output for a set
    * of independent (x) data.
    */
  EXPORT STREAMED DATASET(t_Tensor) Predict(
              STREAMED DATASET(t_Tensor) xDat,
              UNSIGNED4 model) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import numpy as np
    import traceback as tb
    mod = modcache['default']
    try:
      def predGen():
        # We need to process the data one slice at a time, so that we can emit
        # slices with the proper wi and sliceId so that the record indexes line up
        # between the supplied x and the returned predictions.
        outSlices = []
        for slice in xDat:
          # Convert one slice to numpy array format.
          xA = Tens2Np([slice])
          node, wi, sliceId, shape, dataType, maxSliceSize, slice_size, \
                    densedat, sparsedat = slice
          # Restore keras / tf context
          with tfSession.as_default():
            with tfSession.graph.as_default():
              predA = mod.predict(xA)
          preds = []
          # We need to derive the max slice size from the ratio of record sizes
          xSize = np.prod(shape[1:])
          ySize = np.prod(predA.shape[1:])
          newMaxSize = int(maxSliceSize * ySize / xSize)
          # Results should be a single slice, but is returned as a list from Np2Tens(...).
          for s in Np2Tens(predA, maxSliceOverride=newMaxSize):
            preds.append(s)
          pred = preds[0]
          # Apply the wi and sliceId of the original x data to the predictions
          y = (node, wi, sliceId, pred[3], pred[4], pred[5], pred[6], pred[7], pred[8])
          # Yield the output slice.
          yield y
        return
      return predGen()
    except:
      # An error occurred during Predict.
      assert 0 == 1, 'Keras Predict error: ' + format_exc('Predict')
      return []
  ENDEMBED;
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
              UNSIGNED4 model) :=
              EMBED(Python: globalscope(globalScope), persist('query'), activity)
      import traceback as tb
      global nodeId, nNodes, maxSliceLen
      global modcache
      global tfHistory
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
        tfHistory = None
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
        del(nodeId)
        del(nNodes)
        del(maxSliceLen)
        del(tfHistory)
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
        return [(nid, 1, 4, tb.format_exc())]
  ENDEMBED; // Shutdown
END; // Keras Module
