/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
/**
  * Module to access the Human Activity Recognition (HAR) dataset.
  */
EXPORT HARDataset := MODULE

  EXPORT Layout := RECORD
    UNSIGNED8 id;
    DECIMAL5_4 one_0;
    DECIMAL5_4 two_0;
    DECIMAL5_4 three_0;
    DECIMAL5_4 four_0;
    DECIMAL5_4 five_0;
    DECIMAL5_4 six_0;
    DECIMAL5_4 seven_0;
    DECIMAL5_4 eight_0;
    DECIMAL5_4 nine_0;
    DECIMAL5_4 one_1;
    DECIMAL5_4 two_1;
    DECIMAL5_4 three_1;
    DECIMAL5_4 four_1;
    DECIMAL5_4 five_1;
    DECIMAL5_4 six_1;
    DECIMAL5_4 seven_1;
    DECIMAL5_4 eight_1;
    DECIMAL5_4 nine_1;
    DECIMAL5_4 one_2;
    DECIMAL5_4 two_2;
    DECIMAL5_4 three_2;
    DECIMAL5_4 four_2;
    DECIMAL5_4 five_2;
    DECIMAL5_4 six_2;
    DECIMAL5_4 seven_2;
    DECIMAL5_4 eight_2;
    DECIMAL5_4 nine_2;
    DECIMAL5_4 one_3;
    DECIMAL5_4 two_3;
    DECIMAL5_4 three_3;
    DECIMAL5_4 four_3;
    DECIMAL5_4 five_3;
    DECIMAL5_4 six_3;
    DECIMAL5_4 seven_3;
    DECIMAL5_4 eight_3;
    DECIMAL5_4 nine_3;
    DECIMAL5_4 one_4;
    DECIMAL5_4 two_4;
    DECIMAL5_4 three_4;
    DECIMAL5_4 four_4;
    DECIMAL5_4 five_4;
    DECIMAL5_4 six_4;
    DECIMAL5_4 seven_4;
    DECIMAL5_4 eight_4;
    DECIMAL5_4 nine_4;
    DECIMAL5_4 one_5;
    DECIMAL5_4 two_5;
    DECIMAL5_4 three_5;
    DECIMAL5_4 four_5;
    DECIMAL5_4 five_5;
    DECIMAL5_4 six_5;
    DECIMAL5_4 seven_5;
    DECIMAL5_4 eight_5;
    DECIMAL5_4 nine_5;
    DECIMAL5_4 one_6;
    DECIMAL5_4 two_6;
    DECIMAL5_4 three_6;
    DECIMAL5_4 four_6;
    DECIMAL5_4 five_6;
    DECIMAL5_4 six_6;
    DECIMAL5_4 seven_6;
    DECIMAL5_4 eight_6;
    DECIMAL5_4 nine_6;
    DECIMAL5_4 one_7;
    DECIMAL5_4 two_7;
    DECIMAL5_4 three_7;
    DECIMAL5_4 four_7;
    DECIMAL5_4 five_7;
    DECIMAL5_4 six_7;
    DECIMAL5_4 seven_7;
    DECIMAL5_4 eight_7;
    DECIMAL5_4 nine_7;
    REAL4 class_;
  END; // Layout

  SHARED RandomExtended:= RECORD(Layout)
    UNSIGNED rnd;
  END;

  train_0 := DATASET('~.::hartrain.csv',Layout,CSV(heading(0),separator(',')));


  train_1:= PROJECT(train_0, TRANSFORM(RandomExtended, SELF.rnd := RANDOM(), SELF := LEFT));
  train_2 := SORT(train_1,rnd);
  /**
    * Retrieve the HAR Training data
    */
  EXPORT train := PROJECT(train_2, Layout);

  test_0 := DATASET('~.::hartest.csv',Layout,CSV(heading(0),separator(',')));

  test_1:= PROJECT(test_0, TRANSFORM(RandomExtended, SELF.rnd := RANDOM(), SELF := LEFT));
  test_2 := SORT(test_1,rnd);
  /**
    * Retrieve the HAR Test data
    */
  EXPORT test := PROJECT(test_2, Layout);

END; // Module HARDataset
