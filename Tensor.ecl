/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT Python3 AS Python;
IMPORT ML_Core AS core;
IMPORT core.Types AS mlTypes;
IMPORT Std.System.Thorlib;
IMPORT Std.System.Log AS Syslog;
node := Thorlib.node();
nNodes := Thorlib.nodes();
NumericField := mlTypes.NumericField;

/**
  * ECL Tensor Module.
  *
  * <p>Overview:
  * <p>Tensor datasets provide an efficient way to store, distribute,
  * and process N-Dimensional data.
  * Tensors represent an N dimensional array.  They can represent data of from
  * 0 dimensions (scalar), 1 dimension (vector), 2 dimensions (matrix), or
  * up to a high number of dimensions.
  *
  * <p>Tensors are typed -- the module currently only supports REAL4 type
  * Tensors, but is set up to accomodate other data types in the future.
  * The Tensor.R4 submodule is used to manage REAL4 type Tensors.
  *
  * <p>Two main record types are defined for use with Tensors:<ul>
  * <li> TensorData is used to define the content of a Tensor.  This is a
  * sparse data format -- each record represents one cell of the Tensor.</li>
  * <li> t_Tensor is used to define the Tensor's metadata such as it's N
  * dimensional shape, its type, etc.  It manages the tensor as a series
  * of slices (i.e. partitions) with the data packed into the slices in
  * either sparse or dense form, depending on the nature of the data.</li></ul>
  *
  * <p>A Tensor is created by calling the MakeTensor(...) function with the
  * appropriate meta-data and a TensorData dataset.
  *
  * <p>Inversely, the data is read out of a Tensor using the GetData(...)
  * function.
  *
  * <p>Tensor Shape:
  * <p>A Tensor is defined with a shape.  Shapes are given by a set
  *  of integers defining the length of each dimension of the Tensor.
  * For example: shape [4, 3, 2] represents a 4 x 3 x 2 tensor.
  * Record-oriented Tensors may have the first shape component
  * unspecified.  Zero is used to indicated that the index is
  * unspecified.  For example: a shape of [0, 5, 8, 4] specifies
  * a Tensor with an unspecified number of rows, each with a
  * 3 dimensional shape [5, 8, 4].
  *
  * <p>Distribution Modes:
  * <p>Tensors have 2 distribution modes:<ul>
  * <li>Distributed -- The slices are distributed across the nodes of the
  *   cluster.</li>
  * <li>Replicated -- All slices are present on all nodes (for local
  *   operations on each node).</li></ul>
  *
  * <p>Tensor Lists:
  * <p>A t_Tensor dataset also allows for multiple tensors of different
  * shapes to be stored in a single dataset.  The work item (wi) field
  * of the Tensor is used to distinguish between the different Tensors.
  * A Tensor with multiple work items is considered an ordered list of
  * Tensors.
  *
  * <p>Tensor Data Types:
  * <p>At some point, we will support Tensors of different data types
  * such as REAL4, REAL8, INTEGER4, INTEGER8, and STRING.   This release,
  * however, supports only REAL4 type tensors.  The methods operating on
  * these tensors are found in the R4 (i.e. REAL4) submodule.  Future
  * versions will add more submodules for different tensor types.
  *
  * <p>The dat module (e.g. Tensor.R4.dat) provides methods for packing
  * and unpacking scalar, vector, and matrix data.  These methods allow,
  * for example, a Tensor of shape [2,3,3] to be built by packing
  * two 3 x 3 matrices into a Tensor.
  *
  * <p>EXAMPLES:
  * <pre>
  * // Scalar (0-D)
  * tensDatScalar := Tensor.R4.dat.fromScalar(3.14159); // 0D (Scalar) Tensor data
  * // Vector (1-D)
  * tensDatVector := Tensor.R4.dat.fromVector([.013, .015, -.312, 0, 1.0]); // 1D (Vector) Tensor data
  * // Matrix (2-D)
  * tensDatMatrix := Tensor.R4.dat.fromMatrix(myNF); // 2D (Matrix) Tensor data
  * // N-D Tensor
  * tensDat := DATASET([{[1,1,1,1], .01},
  *                      {[5,2,111,3], .02}], Tensor.R4.TensDat); // 4D (nD) Tensor data
  * </pre>
  */
EXPORT Tensor
 := MODULE
  /**
    * @internal
    */
  EXPORT t_Indexes := SET OF UNSIGNED4;
  /**
    * @internal
    */
  EXPORT t_TensType := ENUM(UNUSED=0, R4=1, R8=2, I4=3, I8=4); // Data type of cell
  /**
    * @internal
    */
  EXPORT t_Scope := ENUM(UNUSED=0, DIST=1, REPL=2);
  /**
    * @internal
    */
  EXPORT t_SliceId := UNSIGNED4;
  /**
    * @internal
    */
  EXPORT t_WorkItem := UNSIGNED4;

  // The maximum slice size allowed (in bytes)
  /**
    * @internal
    */
  EXPORT MAX_SLICE := 250000;
  /**
    * Calculate the total (Dense) number of cells in the t_Tensor, given a shape
    * vector.
    */
  SHARED UNSIGNED8 cellCount(t_Indexes shape):= EMBED(C++)
    uint32_t * pos = (uint32_t *)shape;
    uint32_t cnt = lenShape / sizeof(uint32_t);
    uint64_t tot = 1;
    for (uint32_t i = 0; i < cnt; i++)
    {
      uint32_t dimsize = *pos;
      tot *= dimsize;
      pos ++;
    }
    return tot;
  ENDEMBED;
  // Calculate an optimal slice size for this tensor based on its shape, distribution,
  // and the number of nodes in the cluster.
  /**
    * @internal
    */
  EXPORT UNSIGNED4 calcSliceSize(UNSIGNED8 dataSize, UNSIGNED4 elemSize, t_Indexes shape, BOOLEAN isDistributed, UNSIGNED2 breakAtIndex) := FUNCTION
    UNSIGNED4 calcBreakSize(t_Indexes shape, UNSIGNED4 elemSize, UNSIGNED2 breakAtIndex) := EMBED(Python)
      import numpy as np
      if breakAtIndex == 0:
        return 1
      outCount = int(np.prod(shape[breakAtIndex:]))
      outSize = outCount * elemSize
      return outSize
    ENDEMBED;
    // If this is a distributed tensor, then make sure we size the slices such that
    // there is at least one per node.
    minPartitions := IF(isDistributed, nNodes, 1);
    // Breaksize is the smallest unit we need to keep together
    breakSize := calcBreakSize(shape, elemSize, breakAtIndex);
    //maxPartSize := MIN((dataSize / minPartitions + breakSize - 1) div breakSize * breakSize, (MAX_SLICE div breakSize) * breaksize);
    //sliceSize := maxPartSize;
    nSlices := ROUNDUP(dataSize / MAX_SLICE / minPartitions ) * minPartitions;
    sliceSize := ROUNDUP(dataSize / nSlices / breakSize) * breakSize;
    return sliceSize;
  END;
  /**
    * REAL4 tensor type attributes
    *
    */
  EXPORT R4 := MODULE
    /** REAL4 Tensor Data Format.
      * <p>Note: This is sparse format, and any cells not supplied
      * are assumed to be zero.
      *
      * @field indexes -- the N-dimensional index of this tensor cell
      * @field value -- the numeric value of this tensor cell.
      */
    EXPORT TensData := RECORD
      t_Indexes indexes;
      REAL4 value;
    END;
    /**
      * Record format for the sparseData child dataset within a Tensor
      *
      * @field offset The offset within the tensor slice.
      * @field value The value at the given offset within the tensor slice.
      *
      */
    EXPORT t_SparseDat := RECORD
      UNSIGNED4 offset;
      REAL4 value;
    END;
    /**
      * Record format for a REAL4 valued Tensor slice.
      *
      * <p>Tensors are stored as a Dataset of Tensor slices.
      * Each slice contains Tensor metadata (e.g. shape, dataType),
      * as well as the tensor data elements within the slice.
      * Slices can be densely packed or sparsely packed depending
      * on the density of the source data.
      * @field nodeId The node number on which this slice currently
      *    resides.
      * @field wi The work-item allows a list of tensors to be stored
      *   within a single dataset.  Wi of 1 indicates the first tensor
      *   in the list, 2 for the second, etc.
      * @field sliceId The id of this tensor slice.  Each tensor is
      *   represented as 1 or more slices.  Each tensor in a tensor
      *   list can have the same sliceIds.
      * @field shape The shape of the tensor (e.g. [10, 20, 5]).
      * @field dataType The data type for each cell of the tensor.
      * @field maxSliceSize The size of a full slice for this tensor.
      * @field sliceSie The size of this slice.  Slices 1 - (N-1) will
      *   full slices, while slice N may have less than the
      *   maxSliceSize data.
      * @field denseDat A packed block of REAL4 values representing the
      *    linearized data within this slice.
      * @field sparseDat A child dataset for storeing sparse data as
      *   a set of local offset and value pairs.  Note: Only denseData
      *   or sparseData are used for any slice.  The other will be
      *   empty.
      */
    EXPORT t_Tensor := RECORD
      UNSIGNED4 nodeId;
      t_WorkItem wi;
      t_SliceId sliceId;
      t_Indexes shape;
      t_TensType dataType;
      UNSIGNED4 maxSliceSize;
      UNSIGNED4 sliceSize;
      SET OF REAL4 denseData;
      DATASET(t_SparseDat) sparseData;
    END;
    /**
      * Submodule for manipulating TensorData.
      */
    EXPORT dat := MODULE
      /**
        * Create tensor data from a scalar.
        * <p>The scalar will be placed at the "atIndex" in the
        * tensor.
        *
        * <p>Example:
        *   tdat := t_Tensor.R4.dat.fromScalar('3.14159', [1,3,1]);
        *    // The cell will be placed at index [1, 3, 1]
        *
        * @param value The value of the tensor cell at atIndex.
        * @parm atIndx The index of the cell being defined.
        * @return A TensData dataset with one record.
        */
      EXPORT DATASET(TensData) fromScalar(REAL4 value, t_Indexes atIndx = []) := FUNCTION
        outDat := DATASET([{atIndx, value}], TensData);
        RETURN outDat;
      END;
      /**
        * Create tensor data from a vector.
        * <p>The elements of the array will be placed under "atIndx".
        * The first element will be at [atIndx, 1], and the Nth will
        * be at [atIndx, N].
        *
        * <p>Example:
        *   tdat := t_Tensor.R4.dat.fromVector([.1, .2, -.1, -.2], [1, 3]);
        *    // The first element (.1) will be at index [1, 3, 1].
        *
        * @param vec A set of numbers representing the value of the vector.
        * @param atIndx The index under which to place the vector.
        * @return A TensData dataset with length the same as the vector.
        */
      EXPORT DATASET(TensData) fromVector(SET OF REAL4 vec, t_Indexes atIndx = []) := FUNCTION
        outDat := DATASET(COUNT(vec), TRANSFORM(TensData,
                                                      SELF.indexes := atIndx + [COUNTER],
                                                      SELF.value := vec[COUNTER]));
        RETURN outDat;
      END;
      /**
        * Create tensor data from a NumericField matrix.
        * <p>The elements of the matrix will be placed at:
        * [atIndx, id, number], where id and number are the
        * row and column indexes for each matrix cell.
        *
        * <p>Note:  The work-item (wi) field of the NF matrix
        * is ignored, so multiple work-items should not be used in the
        * input matrix.
        *
        * <p>Example:
        *   tdat := t_Tensor.R4.dat.fromMatrix(myNumericFieldDS, [3,5,2]);
        *   // The first element of the matrix will be at: [3,5,2,1,1].
        *
        * @param mat A ML_Core.NumericField dataset representing the matrix
        *        to be added.
        * @param atIndx The index under which to place this matrix in the
        *       tensor data.
        * @return A TensorData dataset with length the same as the NumericField
        *       data passed in.
        * @see ML_Core.Types.NumericField
        */
      EXPORT DATASET(TensData) fromMatrix(DATASET(NumericField) mat, t_Indexes atIndx = []) := FUNCTION
        outDat := PROJECT(mat, TRANSFORM(TensData,
                                        SELF.indexes := atIndx + [LEFT.id, LEFT.number],
                                        SELF.value := LEFT.value), LOCAL);
        RETURN outDat;
      END;
      /**
        * Extract a scalar from a position within the Tensor data.
        * <p>Note: If the tensor shape has 5 indexes, then fromIndex
        * should be 5 long, as the scalar is extracted from the
        * actual tensor cell.
        *
        * <p>Example:
        *   REAL4 val := toScalar(myt_TensorDat, [1,3]);
        *   // Extract a cell from position [1,3] of a 2-D tensor.
        *
        * @param tens A TensData dataset from which to extract.
        * @param fromIndx The index from which to extract the cell value.
        * @return The extracted value as a REAL4.
        */
      EXPORT REAL4 toScalar(DATASET(TensData) tens, t_Indexes fromIndx = []) := FUNCTION
        recs := tens(Indexes = fromIndx);
        val := recs[1].value;
        RETURN val;
      END;
      /**
        * Extract a vector of values from a TensData dataset.
        * <p>If the tensor shape has N terms, then the fromIndx
        * should contain N-1 terms.  It will return the cells:
        * [fromIndx, 1] through [fromIndx, M], where M is the last
        * shape term.
        * <p>The data is returned as a NumericField matrix with
        * a single row (i.e. id = 1).  This is used rather than
        * a SET to allow for sparse data.  Only non-zero cells
        * are returned.  The number field indicates the position
        * within the vector.
        * <p>Example:
        *   DATASET(NumericField) vec := toVector(myt_TensorDat, [5,2]);
        *   // Extract a vector from [5,2] in the 3-D tensor data.
        *
        * @param tens The TensorData dataset from which to extract the vector.
        * @param fromIndex the index from which to extract.
        * @return A vector as a single row of a NumericField matrix.
        * @see ML_Core.Types.NumericField
        */
      EXPORT DATASET(NumericField) toVector(DATASET(TensData) tens, t_Indexes fromIndx = []) := FUNCTION
        NumericField td_to_nf(TensData t) := TRANSFORM
          prefixSize := COUNT(fromIndx);
          suffix := t.indexes[prefixSize+1.. ];
          SELF.id := 1;
          SELF.number := suffix[1];
          SELF.wi := 1;
          SELF := t;
        END;
        prefixSize := COUNT(fromIndx);
        filter := tens.indexes[..prefixSize] = fromIndx;
        outCells := tens(filter);
        outNF := PROJECT(outCells, td_to_nf(LEFT));
        return outNF;
      END;
      /**
        * Extract a matrix of values from a TensData dataset.
        * <p>If the tensor shape has N terms, then the fromIndx
        * should contain N-2 terms.  It will return the cells:
        * [fromIndx, 1, 1] through [fromIndx, K, M], where K is
        * the second to last shape term and M isthe last
        * shape term.
        *
        * <p>Example:
        *   myNF := toNumericField(myt_TensorDat, [3,11]);
        *   // Extract a matrix from a 4-D tensor data dataset.
        *
        * @param tens The TensorData dataset from which to extract.
        * @param fromIndx The index from wich to extract the matrix.
        * @return A matrix in NumericField format.
        * @see ML_Core.Types.NumericField
        */
      EXPORT DATASET(NumericField) toMatrix(DATASET(TensData) tens, t_Indexes fromIndx = []) := FUNCTION
        NumericField td_to_nf(TensData t) := TRANSFORM
          prefixSize := COUNT(fromIndx);
          suffix := t.indexes[prefixSize+1.. ];
          SELF.id := suffix[1];
          SELF.number := suffix[2];
          SELF.wi := 1;
          SELF := t;
        END;
        prefixSize := COUNT(fromIndx);
        filter := tens.indexes[..prefixSize] = fromIndx;
        outCells := tens(filter);
        outNF := PROJECT(outCells, td_to_nf(LEFT));
        return outNF;
      END;
    END; // dat

    /**
      * Replicate the Tensor Slices to all nodes of the cluster.
      * <p>This is used to provide a copy of the Tensor on each node
      * of the cluster.
      *
      * @param tens A t_Tensor dataset to be replicated.
      * @return A replicated t_Tensor dataset.  If the original dataset
      *   contained N slices, the new dataset will contain N x nNodes
      *   slices.
      */
    EXPORT DATASET(t_Tensor) Replicate(DATASET(t_Tensor) tens) := FUNCTION
      tensD := DISTRIBUTE(tens, ALL);
      tensP := PROJECT(NOCOMBINE(tensD), TRANSFORM(RECORDOF(LEFT),
                                              SELF.nodeId := node,
                                              SELF := LEFT), LOCAL);
      tensS := SORT(tensP, wi, sliceId, LOCAL);
      RETURN tensS;
    END;
    /**
      * Internal Function to convert Tensor Data into a packed set
      * of Tensor Slices.
      */
    SHARED STREAMED DATASET(t_Tensor) makeSlices(
                                  STREAMED DATASET(TensData) contents,
                                  UNSIGNED4 wi,
                                  t_Indexes shape,
                                  t_Indexes adjShape,
                                  UNSIGNED4 dtype,
                                  UNSIGNED4 elemSize,
                                  UNSIGNED4 slicesize) :=
                                          EMBED(Python: activity)
      import numpy as np
      import traceback as tb
      try:
        tss = np.prod([x for x in adjShape]) # Total Shape size
        slices = divmod(tss, slicesize)[0] + 1
        sliceId = 0
        # Calculate the size of each indexs contents
        indxSizes = []
        for i in range(len(shape)):
          if i < len(shape) - 1:
            indxSizes.append(int(np.prod(shape[i+1:])))
          else:
            indxSizes.append(1)
        # Function to calculate an index given a flat (zero based) position
        # Indexes are 1 based
        def calcIndxAt(pos):
          indx = []
          remainder = pos
          for i in range(len(shape)):
            val, remainder = divmod(remainder, indxSizes[i])
            indx.append(int(val+1))
          return indx
        # Function to calculate a position (zero based) given a 1-based index
        def calcPosAt(indx):
          pos = 0
          for i in range(len(indxSizes)):
            pos += (indx[i]-1) * indxSizes[i]
          return int(pos)
        def makeDense(datTuples, datSize):
          outdat = [0.0 for d in range(datSize)]
          for tup in datTuples:
            offset, val = tup
            outdat[offset] = val
          return outdat
        sliceId = -1
        sliceDat = []
        for rec in contents:
          indx, val = rec
          pos = calcPosAt(indx)
          thisSliceId, offset = divmod(pos, slicesize)
          if thisSliceId != sliceId:
            # Time to emit the previous slice
            if len(sliceDat) > 0:
              datRemaining = int(tss - sliceId * slicesize)
              datSize = min([datRemaining, slicesize])
              if len(sliceDat) * (elemSize + 4) < slicesize * elemSize:
                # Use sparse form
                outrec = (0, wi, sliceId + 1, shape, dtype, slicesize, datSize, [], sliceDat)
              else:
                # Use dense form
                outrec = (0, wi, sliceId + 1, shape, dtype, slicesize, datSize, makeDense(sliceDat, datSize), [])
              yield outrec
            outdat = [0.0 for d in range(slicesize)]
            if thisSliceId >= slices:
              break
            sliceId = thisSliceId
            sliceDat = []
          sliceDat.append((offset, val))
        if len(sliceDat) > 0:
          datRemaining = int(tss - sliceId * slicesize)
          datSize = min([datRemaining, slicesize])
          if len(sliceDat) * (elemSize + 4) < slicesize * elemSize:
            # Use sparse form
            outrec = (0, wi, sliceId + 1, shape, dtype, slicesize, datSize, [], sliceDat)
          else:
            # Use dense form
            outrec = (0, wi, sliceId + 1, shape, dtype, slicesize, datSize, makeDense(sliceDat, datSize), [])
          yield outrec
      except:
        assert 0 == 1, 'Tensor.MakeSlices: ' + tb.format_exc()
    ENDEMBED;
    /**
      * Internal function to extract TenosrData from a t_Tensor dataset.
      */
    SHARED STREAMED DATASET(TensData) extractData(STREAMED DATASET(t_Tensor) tens) := EMBED(Python:activity)
      import numpy as np
      import traceback as tb
      try:
        # Function to calculate an index given a flat (zero based) position
        # Indexes are 1 based
        shape = None
        indxSizes = []
        def calcIndxAt(pos):
          indx = []
          remainder = pos
          for i in range(len(shape)):
            ix, remainder = divmod(remainder, indxSizes[i])
            indx.append(int(ix + 1))
          return indx
        for rec in tens:
          nodeId, wi, sliceId, shape, datatype, maxslicesize, slicesize, densedat, sparsedat = rec
          if sliceId == 1:
            sliceSize = slicesize
          if not indxSizes:
            for i in range(len(shape)):
              indxSizes.append(int(np.prod(shape[i+1:])))
          slicePos = maxslicesize * (sliceId - 1) # base position of slice
          if not densedat:
            # Do sparse decoding
            for item in sparsedat:
              offset, val = item
              indx = calcIndxAt(slicePos + offset)
              yield((indx, val))
          else:
            # Do dense decoding
            for v  in range(slicesize):
              val = densedat[v]
              if abs(val) > .000000001: # Only output non-zero entries
                                        #  +/- 10^-9 is considered zero
                indx = calcIndxAt(slicePos + v) # @ the position in the flattened array
                yield((indx, val))
      except:
        assert 0 == 1, 'Tensor.extractData: ' + tb.format_exc()

    ENDEMBED;

    /**
      * Replace any dimension shapes of zero (unspecified) with the maximum index
      * of that dimension in the data.
      */
    SHARED adjustShape(t_Indexes shape, DATASET(TensData) contents) := FUNCTION
      // If the cell count computes as zero, it means we have at least one unspecified
      // dimension.  If not, there is no need to adjust
      needsAdjusting := cellCount(shape) = 0;
      md := TABLE(contents, {max1 := MAX(GROUP, indexes[1]), max2 := MAX(GROUP, indexes[2]),
                    max3 := MAX(GROUP, indexes[3]), max4 := MAX(GROUP, indexes[4]), max5 := MAX(GROUP, indexes[5])})[1];
      t_Indexes maxDims0 := [md.max1, md.max2, md.max3, md.max4, md.max5];
      maxDims := maxDims0[1 .. COUNT(shape)];

      t_Indexes calcFinal(t_Indexes shape, t_Indexes maxdims) := EMBED(C++)
        uint32_t cnt = lenShape / sizeof(uint32_t);
        __result = (void *)rtlMalloc(lenShape);
        __lenResult = lenShape;
        __isAllResult = FALSE;
        uint32_t * result = (uint32_t *) __result;
        uint32_t * uShape = (uint32_t *) shape;
        uint32_t * uMaxdims = (uint32_t *) maxdims;
        for (uint32_t i=0; i < cnt; i++)
        {
          if (uShape[i] == 0)
            result[i] = uMaxdims[i];
          else
            result[i] = uShape[i];
        }
      ENDEMBED;
      result := IF(needsAdjusting, calcFinal(shape, maxDims), shape);
      RETURN result;
    END;
    /**
      * Calculate the hierarchical size of each index in the Tensor's
      * shape.  For example, if the Tensor is 3-D [12, 5, 2], the
      * size of the first index would be 5 x 2 = 10, while the size
      * of the second index would be 2.  This is used during
      * flattening and unflattening of the Tensor.
      */
    SHARED t_Indexes calcIndexSizes(t_Indexes shape) := EMBED(Python)
      import numpy as np
      # Calculate the size of each indexes contents
      indxSizes = []
      for i in range(len(shape)):
        if i < len(shape) - 1:
          indxSizes.append(int(np.prod(shape[i+1:])))
        else:
          indxSizes.append(1)
      return indxSizes
    ENDEMBED;
    /**
      * Calculate the node id on which a given wi, sliceId combination should be
      * allocated.
      */
    SHARED UNSIGNED4 calcNodeId(UNSIGNED4 wi, UNSIGNED4 sliceId, UNSIGNED4 nSlices) := FUNCTION
      // Allocate slices sequentially to nodes
      slicesPerNode := nSlices / nNodes;
      relNode := (sliceId - 1) div slicesPerNode;
      nodeId := relNode % nNodes;
      return nodeId;
    END;
    /**
      * Calculate the sliceId in which a given Tensor cell will reside.
      */
    SHARED UNSIGNED4 calcSliceId(UNSIGNED recNum, UNSIGNED4 recSize, UNSIGNED4 sliceSize) := FUNCTION
      pos := (recNum - 1) * recSize;
      RETURN (pos DIV sliceSize) + 1;
    END;
    /**
      * Optimized version of calcNodeId(wi, calcSliceId(...), nSlices)
      * Because of the number of times this is called, we need to optimize as much as possible,
      * even at the cost of clarity.
      */
    SHARED UNSIGNED4 calcNodeId2(UNSIGNED4 wi, UNSIGNED4 nSlices, UNSIGNED4 sliceSize, UNSIGNED recNum, UNSIGNED4 recSize) := FUNCTION
      sliceIdZ :=((recNum -1) * recSize) DIV sliceSize; // Zero based
      relNode := sliceIdZ DIV (nSlices / nNodes);
      //nodeId := (wi - 1) + relNode % nNodes;
      nodeId := relNode % nNodes; // Temporarily disable spreading by wi
      RETURN nodeId;
    END;

    /**
      * Make a Tensor from a set of TensorData and
      * some meta-data.
      * <p>Tensors may be replicated (e.g. copied locally to each node), or
      * distributed (slices spread across nodes).
      * @param shape The desired shape of the Tensor (e.g. [10, 5, 2]).
      * @param contents Dataset of TensData representing the contents of the Tensor.
      *         If omitted, the tensor will be empty (i.e. all zeros).
      * @param replicated True if this tensor is to be replicated to all nodes.
      *            Default = False (i.e. distributed).
      * @param wi Work-item.  This field allows multiple Tensors to be stored
      *           in the same dataset.  Default = 1.  This field should always
      *            be 1 for a single Tensor dataset.  For a Tensor list, wi
      *            should always go from 1 to nTensors.
      * @param forceMaxSliceSize If non-zero, it will override the default sizing
      *        of slices.  Needed internally, but should always use the default
      *        (0) for external uses.
      * @return A dataset of t_Tensor representing the Tensor object.
      */
    EXPORT DATASET(t_Tensor) MakeTensor(t_Indexes shape,
                                DATASET(TensData) contents = DATASET([], TensData),
                                BOOLEAN replicated = FALSE,
                                UNSIGNED4 wi = 1,
                                UNSIGNED4 forceMaxSliceSize = 0) := FUNCTION
      isDistributed := NOT replicated; // If not replicated then distributed
      // If the first term of the shape is 0 (uspecified), then the data is record oriented
      // vs block oriented.  In that case, set breakAtIndex to 1, to make sure that the
      // slices don't span record boundaries.  We assume that there is no need for a
      // multi-dimensional record id.  If there is a need for that, we will have to
      // expand this logic.
      breakAtIndex := IF(shape[1] = 0, 1, 0);
      adjShape := adjustShape(shape, contents);
      totalCount := cellCount(adjShape);
      elemSize := 4;
      totalSize := totalCount * elemSize;
      sliceSize0 := calcSliceSize(totalSize, elemSize, adjShape, isDistributed, breakAtIndex);
      sliceSize := IF(forceMaxSliceSize > 0, forceMaxSliceSize, sliceSize0);
      sliceElems := sliceSize / elemSize;
      nSlices := ROUNDUP(totalSize / sliceSize);
      indxSizes := calcIndexSizes(shape);
      recSize := indxSizes[1];
      contentsD := DISTRIBUTE(contents, calcNodeId2(wi, nSlices, sliceElems, indexes[1], recSize));
      contentsDS := SORT(NOCOMBINE(contentsD), indexes[1], LOCAL);
      slices0 := makeSlices(contentsDS, wi, shape, adjShape, t_TensType.R4, elemSize, sliceElems);
      // If not replicated, slices are already correctly distributed (i.e. by wi and sliceId)
      slices1 := IF(replicated, Replicate(slices0), PROJECT(slices0, TRANSFORM(RECORDOF(LEFT),
                                                            SELF.nodeId := node,
                                                            SELF := LEFT), LOCAL));
      slices := SORT(slices1, sliceId, LOCAL);
      RETURN slices;
    END;
    /**
      * Restore a replicated Tensor to a single distributed Tensor
      */
    SHARED DATASET(t_Tensor) deReplicate(DATASET(t_Tensor) tens) := FUNCTION
      nSlices := COUNT(tens);
      maxSlice := MAX(tens, sliceId);
      slicesPerNode := nSlices / nNodes;
      wi := tens[1].wi;
      derep := tens(nodeId = calcNodeId(wi, sliceId, nSlices));
      // Only de-rep if it is a replicated tensor, otherwise bad things can happen.
      outTens := IF(nSlices > maxSlice, derep, tens);
      return outTens;
    END;
    /**
      * Extract the data from a tensor and return it in sparse TensData format.
      * <p>This is essentially the inverse of the MakeTensor(...) method.
      *
      * @param tens The t_Tensor dataset from which to extract the data
      * @return TensData dataset of non-zero tensor data (sparse form).
      */
    EXPORT DATASET(TensData) GetData(DATASET(t_Tensor) tens) := FUNCTION
      // Get rid of any replicated records and leave distributed by wi and sliceId
      dereplicated := deReplicate(tens);
      dat := extractData(dereplicated);
      RETURN dat;
    END;
    /**
      * Convert sparse data to dense data
      */
    SHARED SET OF REAL4 getDense(t_Tensor slice) := FUNCTION
      SET OF REAL4 makeDense(DATASET(t_SparseDat) sparse, UNSIGNED4 datsize) := EMBED(Python)
        outList = [0.0 for i in range(datsize)]
        for rec in sparse:
          indx, val = rec
          assert indx < datsize, 'Tensor.makeDense: indx = ' + str(indx) + ', datsize = ' + str(datsize)
          outList[indx] = val
        return outList
      ENDEMBED;
      dense := IF(EXISTS(slice.denseData), slice.denseData, makeDense(slice.sparseData, slice.sliceSize));
      RETURN dense;
    END;
    /**
      * Determine whether this Tensor slice is more efficiently stored in a sparse or
      * dense form.  Initilizes sparseData or denseData within the slice, as appropriate.
      */
    SHARED t_Tensor compressIfNeeded(t_Tensor slice, SET OF REAL4 newdense = []) :=
            EMBED(Python)
      import traceback as tb
      try:
        nodeid, wi, sliceid, shape, datatype, maxSliceSize, slicesize, densedata, sparsedata = slice
        sparsecount = 0
        sparseData = []
        assert len(newdense) == slicesize, 'compressIfNeeded: Data size does not match sliceSize -- ' + \
                      str(len(newdense)) + ', ' + str(slicesize) + ', ' + str(wi) + ', ' + str(sliceid)
        denseData = newdense
        for i in range(slicesize):
          val = denseData[i]
          if abs(val) > .000000001:
            sparsecount += 1
            sparseData.append((i, val))
        if sparsecount * (8) < slicesize * 4:
          # Sparse encoding
          denseData = []
        else:
          # Leave dense
          sparseData = []
        return (nodeid, wi, sliceid, shape, datatype, maxSliceSize, slicesize, denseData, sparseData)
      except:
        assert 0 == 1, 'Tensor.compressIfNeeded: ' + tb.format_exc()
    ENDEMBED;
    /**
      * Element wise addition of two sets of dense data.
      */
    SHARED SET OF REAL4 addSliceData(SET OF REAL4 d1, SET OF REAL4 d2) := EMBED(Python)
      import numpy as np
      assert len(d1) == len(d2), 'addSliceData: sizes do not match. ' + str(len(d1)) + ', ' + str(len(d2))
      d1A = np.array(d1)
      d2A = np.array(d2)
      return list(d1A + d2A)
    ENDEMBED;
    /**
      * Determine if two shapes are compatible for re-shaping.
      */
    SHARED BOOLEAN areShapesCompatible(t_Indexes currShape, t_Indexes newShape) := EMBED(Python)
      import numpy as np
      import traceback as tb
      try:
        if currShape[0] == 0 and newShape[0] == 0:
          # It is a record-oriented tensor.  The product of the 2nd through Nth terms
          # must be equeal.
          return int(np.prod(currShape[1:])) == int(np.prod(newShape[1:]))
        elif currShape[0] > 0 and newShape[0] > 0:
          # Block oriented tensor.  The product of all terms must be equal.
          return int(np.prod(currShape)) == int(np.prod(newShape))
        else:
          # One is zero and the other non-zero.  Not compatible.
          return False
      except:
        assert 0 == 1, 'areShapesCompatible Error: ' + tb.format_exc()
    ENDEMBED;
    /**
      * Reshape a tensor to a new compatible shape.
      *
      * <p>Returns a new tensor with the desired shape.
      * <p>If the shapes were not compatible, an empty tensor is returned.
      *
      * @param tens The tensor to be reshaped.
      * @param newShape The desired new shape.
      * @return A new tensor with the desired shape, if the shapes were
      *      compatible.  Otherwise, an empty tensor.
      */
    EXPORT Reshape(DATASET(t_Tensor) tens, t_Indexes newShape) := FUNCTION
      currShape := tens[1].shape;
      areCompatible := areShapesCompatible(currShape, newShape);
      newTens := PROJECT(tens, TRANSFORM(RECORDOF(LEFT),
                                  SELF.shape := newShape,
                                  SELF := LEFT), LOCAL);
      empty := DATASET([], t_Tensor);
      result := IF(areCompatible, newTens, empty);
      RETURN result;
    END;
    /**
      * Add two tensors.
      * <p> This performs cell-wise addition of the contents of the two input tensors
      * and returns a new tensor representing the sum of the two tensors.
      * <p> Both tensors must be of the same shape.
      * <p>This function can also add two tensor lists.  Each tensor of
      * list 1 must be of the same shape as the corresponding tensor in
      * list 2.  The lists must also be of the same length.
      * @param t1 The first tensor or tensor list.
      * @param t2 The second tensor or tensor list.
      * @return A new Tensor (DATASET(t_Tensor)) representing t1 + t2.
      */
    EXPORT DATASET(t_Tensor) Add(DATASET(t_Tensor) t1, DATASET(t_Tensor) t2) := FUNCTION
      dense1 := PROJECT(t1, TRANSFORM(RECORDOF(LEFT),
                              SELF.densedata := getDense(LEFT),
                              SELF.sparsedata := DATASET([], t_SparseDat),
                              SELF := LEFT), LOCAL);
      dense2 := PROJECT(t2, TRANSFORM(RECORDOF(LEFT),
                              SELF.densedata := getDense(LEFT),
                              SELF.sparsedata := DATASET([], t_SparseDat),
                              SELF := LEFT), LOCAL);

      tSum := JOIN(dense1, dense2, LEFT.wi = RIGHT.wi AND LEFT.sliceId = RIGHT.sliceId,
                    TRANSFORM(RECORDOF(LEFT),
                          SELF.densedata := addSliceData(LEFT.denseData, RIGHT.denseData),
                          SELF := LEFT), FULL OUTER, LOCAL);
      out := PROJECT(tSum, TRANSFORM(RECORDOF(LEFT),
                        SELF := compressIfNeeded(LEFT, LEFT.denseData)), LOCAL);
      RETURN out;
    END; // Add

    /**
      * @internal
      * Add 2 tensor slices.  This is for internal use only.
      * @param s1 The first tensor slice.
      * @param s2 The second tensor slice.
      * @return A tensor slice containing the element-wise sum of s1
      *           and s2.
      */
    EXPORT t_Tensor AddSlices(t_Tensor s1, t_Tensor s2) := FUNCTION
      dense1 := getDense(s1);
      dense2 := getDense(s2);
      denseSum := addSliceData(dense1, dense2);
      newS := compressIfNeeded(s1, denseSum);
      RETURN newS;
    END; // AddSlices
    /**
      * Get the number of records in a record-oriented Tensor.
      *
      * @param tens The input Tensor.
      * @return The number of records in the distributed tensor.
      */
    EXPORT UNSIGNED GetRecordCount(DATASET(t_Tensor) tens) := FUNCTION
      UNSIGNED4 getRecSize(t_Indexes shape) := EMBED(Python)
        import numpy as np
        recSize = int(np.prod(shape[1:]))
        return recSize
      ENDEMBED;
      shape := tens[1].shape;
      recSize := getRecSize(shape);
      tab1 := TABLE(tens, {totSize := SUM(GROUP, sliceSize)});
      totSize := tab1[1].totSize;
      nRecs := totSize DIV recSize;
      RETURN nRecs;
    END; // GetRecordCount
    /**
      * @internal
      * Internal use only.
      * Aligns a pair of record-oriented tensors such that the same record number
      * of each Tensor will be on the same node.  This prevents different sized
      * records from being distributed differently among the nodes.
      * @param tens A Tensor List with two tensors, A (wi = 1) and B (wi = 2).
      * @return A Tensor List with two new tensors A and B, identified as above.
      *        The returned A and B tensors are aligned.
      **/
    EXPORT DATASET(t_Tensor) AlignTensorPair(DATASET(t_Tensor) tens) := FUNCTION
      // This can be optimized if necessary by custom restructuring
      //  versus use of GetData(...) and MakeTensor(...)
      tA := tens(wi = 1);
      tB0 := tens(wi = 2);
      tB := PROJECT(tB0, TRANSFORM(RECORDOF(LEFT), SELF.wi := 1, SELF := LEFT), LOCAL);
      UNSIGNED recSize(t_Indexes shape) := EMBED(Python)
        import numpy as np
        recSz = np.prod(shape[1:])
        return int(recSz)
      ENDEMBED;
      UNSIGNED maxRecsPerSlice(DATASET(t_Tensor) tens, UNSIGNED recSize) := FUNCTION
        sliceSize := tens[1].maxSliceSize;
        return sliceSize / recSize;
      END;
      Adat := GetData(tA);
      Ashape := tA[1].shape;
      ArecSize := recSize(Ashape);
      ArecsPerSlice := maxRecsPerSlice(tA, ArecSize);
      Bdat := GetData(tB);
      Bshape := tB[1].shape;
      BrecSize := recSize(Bshape);
      BrecsPerSlice := maxRecsPerSlice(tB, BrecSize);
      elemSize := 4;
      AnewSliceSize := BrecsPerSlice * ArecSize * elemSize;
      BnewSliceSize := ArecsPerSlice * BrecSize * elemSize;
      alignedA := MakeTensor(Ashape, Adat, wi := 1, forceMaxSliceSize := AnewSliceSize);
      alignedB0 := MakeTensor(Bshape, Bdat, wi := 1, forceMaxSliceSize := BnewSliceSize);
      // We want the B tensor to be distributed the same as A, so we want to use wi = 1
      // when we do MakeTensor(...), so we project the result to wi = 2 after the MakeTensor,
      // in order to distinguish the two tensors.
      alignedB := PROJECT(alignedB0, TRANSFORM(RECORDOF(LEFT),
                                          SELF.wi := 2, SELF := LEFT), LOCAL);
      // Re-align the smaller tensor to use the same number of records per slice as
      // the larger tensor.  If the tensors happen to be already aligned, we return
      // the original tensors.
      reAligned := IF(ArecsPerSlice < BrecsPerSlice, tA + alignedB,
                    IF(ArecsPerSlice > BrecsPerSlice, alignedA + tB0, tens));
      RETURN reAligned;
    END; // AlignTensorPair
    /**
      * Aligns a list of Tensors (seperated by wi) so that all of the tensors'
      * corresponding records are stored on the same node.
      * This prevents different sized
      * records from being distributed differently among the nodes.
      * <p>In most cases, the inputs and outputs to a neural network during training,
      * and the inputs during prediction should be aligned so that
      * various aspects of the same observation are presented together.
      *
      * @param tens A Tensor List with at least two tensors identified by
      *    sequential work item ids from 1-N.
      * @return A new Tensor List with the same number of tensors as the input
      *    list, with all of the tensors being aligned.
      **/
    EXPORT DATASET(t_Tensor) AlignTensors(DATASET(t_Tensor) tensList) := FUNCTION
      // This can be optimized if necessary by custom restructuring
      //  versus use of GetData(...) and MakeTensor(...)
      elemSize := 4; // REAL4
      UNSIGNED recSize(t_Indexes shape) := EMBED(Python)
        import numpy as np
        recSz = np.prod(shape[1:])
        return int(recSz)
      ENDEMBED;
      itemInfo0 := TABLE(tensList, {wi, shape, maxSliceSize, UNSIGNED recsPerSlice := 0, UNSIGNED recSize := 0}, wi, shape, maxSliceSize);
      itemInfo1 := PROJECT(itemInfo0, TRANSFORM(RECORDOF(LEFT),
                                    SELF.recSize := recSize(LEFT.shape),
                                    SELF.recsPerSlice := LEFT.maxSliceSize / SELF.recSize,
                                    SELF := LEFT), LOCAL);
      itemInfo := SORT(itemInfo1, recsPerSlice, -recSize);
      largestRecItem := itemInfo[1];
      newRecSize := largestRecItem.recSize;
      newRecsPerSlice := largestRecItem.recsPerSlice;
      largestRecWI := largestRecItem.wi;
      numTensors := COUNT(itemInfo);
      DATASET(t_Tensor) adjustTensors(DATASET(t_Tensor) tl, UNSIGNED ctr) := FUNCTION
        // Do one tensor for each loop
        thisTens := tl(wi = ctr);
        thisTensDat := GetData(thisTens);
        thisTensShape := thisTens[1].shape;
        thisMaxSliceSize := newRecsPerSlice * recSize(thisTensShape) * elemSize;
        // We want all the tensors to be aligned the same, so we create the slices with
        // wi of the largestRecItem, and then project to the correct wi.  This is because MakeTensor spreads
        // the wi's across nodes.  By Making all the tensors with the wi of the largestRecItem, we
        // save the need to re-create that largest of the tensors.
        adjTens0 := MakeTensor(thisTensShape, thisTensDat, wi := largestRecWI, forceMaxSliceSize := thisMaxSliceSize);
        adjTens := PROJECT(adjTens0, TRANSFORM(RECORDOF(LEFT), SELF.wi := ctr, SELF := LEFT), LOCAL);
        // If this is the tensor with the largest rec size, don't need to adjust.  Otherwise adjust.
        newTens := IF(ctr = largestRecWI, thisTens, adjTens);
        outTens := tl(wi != ctr) + newTens;
        return outTens;
      END;
      reAligned := LOOP(tensList, numTensors, LEFT.wi >= COUNTER, adjustTensors(ROWS(LEFT), COUNTER));
      RETURN SORT(reAligned, sliceId, LOCAL);
    END; // AlignTensors
  END; // R4
END; // t_Tensor
