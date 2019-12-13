/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Type definitions for use with the GNNI Interface.
  */
EXPORT Types := MODULE
  /**
    * Return structure for call to EvaluateMod.
    * <p>Contains a series of metrics and their values.
    * @field metriId A sequential id to maintain the metrics' order.
    * @field metriName The Keras name identifying the metric.
    * @field value The value of the metric.
    */
  EXPORT metrics := RECORD
    UNSIGNED4 metricId;
    STRING metricName;
    REAL8 value;
  END;
END;
