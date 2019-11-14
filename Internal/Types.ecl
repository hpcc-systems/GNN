/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Internal type definitions for the GNN bundle.
  */
EXPORT Types := MODULE

  // When updating kStrType, Keras.Init() must also be updated with correct enum
  /**
    * Enumerations for the typ field of the KString record
    */
  EXPORT kStrType := ENUM(UNSIGNED4, None=0, layer=1, compile=2, json=3, status=4);
  /**
    * General distributed string passing interface for use with the Keras module.
    */
  EXPORT kString := RECORD
    UNSIGNED4 nodeId := 0;
    UNSIGNED4 id;
    KStrType typ;
    STRING text;
  END;
  /**
    * Distributed input to the Keras Init function.
    */
  EXPORT initParms := RECORD
    UNSIGNED4 nodeId;
    UNSIGNED4 nNodes;
    UNSIGNED4 maxSliceSize;
  END;
  /**
    * Record to hold the losses returned from Keras GetLoss.
    */
  EXPORT losses := RECORD
    REAL8 loss;
  END;
END;
