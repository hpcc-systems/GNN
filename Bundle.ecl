/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT Std;
EXPORT Bundle := MODULE(Std.BundleBase)
  EXPORT Name := 'GNN';
  EXPORT Description := 'Generalized Neural Network Bundle';
  EXPORT Authors := ['HPCCSystems'];
  EXPORT License := 'See LICENSE.TXT';
  EXPORT Copyright := 'Copyright (C) 2019 HPCC Systems®';
  EXPORT DependsOn := ['ML_Core'];
  EXPORT Version := '1.0.0';
  EXPORT PlatformVersion := '7.4.0';
END;
