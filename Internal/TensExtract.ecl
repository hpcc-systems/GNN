IMPORT PYTHON3 as PYTHON;
IMPORT $.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT Std.System.Thorlib;

nodeId := Thorlib.node();
nNodes := Thorlib.nodes();

t_Tensor := Tensor.R4.t_Tensor;

MAX_SLICE := POWER(2, 24);

/**
  * This function is used by GNNI to pull local samples from the X and Y tensors.
  * The result is a new tensor with samples from each local slice of the tensor.
  * Note that this will extract datcount samples from EACH node.  The pos parameter
  * indicates how far into the local tensor slices to start extracting.
  * If there are multiple tensors in the tensor dataset, then extract datcount
  * samples from each one.  If there are multiple tensors in the dataset, then
  * it is essential to align them before calling this function
  * @see Tensor.AlignTensors
  */
EXPORT DATASET(t_Tensor) TensExtract(DATASET(t_Tensor) tens, UNSIGNED pos,
                                    UNSIGNED datcount) := FUNCTION
  // Python embed function to do most of the heavy lifting.
  STREAMED DATASET(t_Tensor) extract(STREAMED DATASET(t_Tensor) tens,
            UNSIGNED pos, UNSIGNED datcount, nodeid, nNodes, maxslice) := EMBED(Python: activity)
    import numpy as np
    import traceback as tb
    maxSliceLen = maxslice
    dTypeDict = {1:np.float32, 2:np.float64, 3:np.int32, 4:np.int64}
    dTypeDictR = {'float32':1, 'float64':2, 'int32':3, 'int64':4}
    dTypeSizeDict = {1:4, 2:8, 3:4, 4:8}
    # Generator Function to convert a numpy array to a tensor.
    def Np2Tens(a, wi=1):
      epsilon = .000000001
      origShape = list(a.shape)
      # For final shape, the first component should be zero to indicate a record-oriented
      # tensor.
      finalShape = [0] + origShape[1:]
      flatA = a.reshape(-1)
      flatSize = flatA.shape[0]
      sliceId = nodeid + 1
      indx = 0
      maxSliceSize = 0
      datType = dTypeDictR[str(a.dtype)]
      elemSize = dTypeSizeDict[datType]
      max_slice = divmod(maxSliceLen, elemSize)[0]
      while indx < flatSize:
        remaining = flatSize - indx
        if remaining >= max_slice:
          sliceSize = max_slice
        else:
          sliceSize = remaining
        #if sliceId == 1:
        #  maxSliceSize = sliceSize
        maxSliceSize = sliceSize
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
            yield (nodeid, wi, sliceId, finalShape, datType, maxSliceSize, sliceSize, [], sparse)
          else:
            # Dense encoding
            yield (nodeid, wi, sliceId, finalShape, datType, maxSliceSize, sliceSize, dat, [])
        sliceId += 1
        indx += sliceSize
        # END OF NP2Tens
    # Generator function to return the extract from a list of tensors
    def getResults():
      try:
        outArray = None
        tshape = []
        sliceNum = 0
        fullSize = 0
        rowSize = 0
        outSize = 0
        startSlice = 0
        startPos = 0
        endSlice = 0
        endPos = 0
        outPos = 0
        currWi = 0
        # If the first shape component is non-zero, then this is a fixed size Tensor
        # and exact positions are important.  If not fixed sized, then we take the
        # records sequentially and don't fill gaps.  We determine size by the actual
        # records received.
        isFixedSize = False
        for rec in tens:
          node, wi, sliceId, shape, dataType, maxSliceSize, slice_size, densedat, sparsedat = rec
          dtype = dTypeDict[dataType]
          if wi != currWi:
            if outArray is not None:
              # New wi.  Output the previous one.
              if outPos < datcount * rowSize:
                # Fewer than requested records available.
                outArray.resize((outPos,))
              # If this is a variable size tensor, reflect that in the numpy array.
              outArray = np.reshape(outArray, tshape)
              # Yield the previous wi's tensor and reset for the new wi.
              yield from Np2Tens(outArray, currWi)
              outArray = None
              tshape = []
            currWi = wi
          if outArray is None:
            # Initialize important information on the first slice.
            # The output shape (tshape).  Note: Only supports record oriented tensors.
            if shape[0] == 0:
              tshape = [-1] + shape[1:] # Make first term -1 for numpy tensor
            else:
              raise Exception('Extract requires record-oriented tensors ' + \
                'that must have a zero first shape component. Shape = ' + str(shape) + '.')
            # Full size of the tensor
            fullSize = np.prod(shape)
            # Is fixed size if the first component of the shape is 0.
            isFixedSize = fullSize != 0
            # Row size is the size of the 2nd - last shape component.
            rowSize = np.prod(shape[1:])
            # Calculate the size to be returned
            outSize = rowSize * datcount
            # Create an array of zeros to hold the output.
            outArray = np.zeros((outSize,), dtype)
            # Figure out which slice and position the desired data starts
            # and ends on
            startSlice, startPos = divmod(pos * rowSize, maxSliceSize)
            endSlice, endPos = divmod((pos + datcount) * rowSize, maxSliceSize)
            # Slice number
            sliceNum = 0
            outPos = 0
          if sliceNum < startSlice or sliceNum > endSlice:
            # The data is found in a later slice or we're already past the end of
            # the data.  We have to keep iterating in the latter case because there
            # might be more wi's.  Skip this record.
            sliceNum += 1
            continue
          if not densedat:
            # Sparse decoding
            dat = np.zeros((slice_size,), dtype)
            for offset, val in sparsedat:
              assert offset < slice_size, 'TensExtract: sparsedat has higher index than the sliceSize = ' + str(offset)
              dat[offset] = dtype(val)
            densedat = dat
          if sliceNum == startSlice and sliceNum == endSlice:
            # Data starts and ends on this slice.
            densedat = densedat[startPos:endPos]
          elif sliceNum == startSlice:
            # Data starts on this slice, but ends on a further one.
            densedat = densedat[startPos:]
          elif sliceNum == endSlice:
            # Data ends on this slice but started previously.
            densedat = densedat[:endPos]
          # Add any data from this slice
          outArray[outPos:outPos + len(densedat)] = densedat
          outPos += len(densedat)
          sliceNum += 1
          # END for
        if sliceNum == 0:
          # No data in the slice.   Return an empty Tensor.
          return []
        if outPos < datcount * rowSize:
          # Fewer than requested records available.
          outArray.resize((outPos,))
        # If this is a variable size tensor, reflect that in the numpy array.
        outArray = np.reshape(outArray, tshape)
        # Yield the final wi's tensor
        yield from Np2Tens(outArray, wi)
      except:
        # Error during extraction.
        assert 0 == 1, 'TensExtract: ' + str(tshape) + ',' + 'currWi = ' + str(currWi) + ', ' + tb.format_exc()
      # END OF getResults()
    return getResults()
  ENDEMBED; // Extract
  RETURN SORT(extract(tens, pos-1, datcount, nodeId, nNodes, MAX_SLICE), wi, sliceId, LOCAL);
END;