IMPORT $ AS GNN;
IMPORT GNN.Tensor;
IMPORT Std.System.Thorlib;

TensDat := Tensor.R4.TensData;
/**
  * Utility module for GNN.  Contains various utility functions for use with GNN.
  */
EXPORT Utils := MODULE
  /** 
    * Method to setup effective node number as per available nodes
    */
  EXPORT INTEGER getEffNodesNumber(nodeNumber) := FUNCTION
    /*
    nodeNumber: default value = 0 (meaning distribute to all nodes)
                if totalAvailableNode<nodeNumber<=0 then fallback to totalAvailableNodes,
    */
    totalAvailableNodes := Thorlib.nodes();
    eNodeNumber := IF(nodeNumber>0, nodeNumber, totalAvailableNodes);
    // clipping eNodeNumber to totalAvailableNodes
    return IF(eNodeNumber<totalAvailableNodes, eNodeNumber, totalAvailableNodes);
  END;
 

  /**
    * Convert Tensor Data to OneHot Encoding.
    * <p>Input is a 1-D tensor data set with the value of each
    * observation being the class.
    * <p>Returns a 2-D TensDat dataset with numClasses being
    * the cardinality of the 2nd dimension.  The value will
    * be 1 for the cell with second dimension corresponding to the
    * class.  All others will be zero.  Since TensDat is a
    * sparse format, all zero cells will be skipped.
    * <p>Note that Classes are 0-based. Class 0 will be at
    * final index = 1.  Class 5 will be at final index = 6.
    * @param classDat A 1-D tensor with the index being the
    * observation number, and the value ((0-(numClasses-1))
    * corresponds to the class label.
    * @param numClasses The number of possible values for
    *     the class variable.
    * @return A 2-D set of TensData one hot encoded.
    * @see Tensor.R4.TensData
    */
  EXPORT DATASET(TensDat) ToOneHot(DATASET(TensDat) classDat, UNSIGNED numClasses) := FUNCTION
    oh := NORMALIZE(classDat, numClasses, TRANSFORM(TensDat,
                                        SELF.indexes := [LEFT.indexes[1], COUNTER],
                                        SELF.value := IF(COUNTER = LEFT.value + 1, 1, skip)
                                        ));
    RETURN oh;
  END;
  /**
    * Convert One Hot encoded 2-D tensor data to class label format.
    * <p>Input is a 2-D One Hot encoded TensDat dataset, as produced
    * by ToOneHot above.
    * <p>Output is a 1-D set of class labels corresponding to the
    * highest value of the One Hot encoded fields for each observation.
    * <p>Note that returned classes are zero based.
    * @param ohTens A one hot encoded 2-D tensor data set.
    * @return A 1-D dataset of TensData with the value of each observation
    *     being the class label.
    * @see Tensor.R4.TensData
    */
  EXPORT DATASET(TensDat) FromOneHot(DATASET(TensDat) ohTens) := FUNCTION
    ohDist := DISTRIBUTE(ohTens, indexes[1]);
    ohSorted := SORT(ohDist, indexes[1], -value, LOCAL);
    ohDedup := DEDUP(ohSorted, indexes[1], LOCAL);
    result := PROJECT(ohDedup, TRANSFORM(TensDat,
                                        SELF.indexes := [LEFT.indexes[1], 1],
                                        SELF.value := LEFT.indexes[2] - 1), LOCAL);
    RETURN result;
  END;
  /**
    * Convert a set of class probabilities to a class label
    * <p>Class probabilities are typically returned from a "softmax"
    * activation function.  This returns the class label associated
    * with the maximum probability label.
    * <p>Note that this function simply calls FromOneHot, which
    * implements this functionality.  Both names are included
    * because it is sometimes more intuitive to think of the operation
    * in different ways.
    * @param td A 2-D tensor data set with a probability for each class.
    * @return A 1-D dataset of TensData with a class label for each observation.
    * @see Utils.FromOneHot
    * @see Tensor.R4.TensData
    */
  EXPORT DATASET(TensDat) Probabilities2Class(DATASET(TensDat) td) := FUNCTION
    RETURN FromOneHot(td);
  END;
END;
