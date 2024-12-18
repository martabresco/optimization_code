\ Model Master Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2.9744e+06 xW[0] + 2.9744e+06 xW[1] + 2.9744e+06 xW[2] + 2.9744e+06 xW[3]
   + 2.9744e+06 xW[4] + 2.9744e+06 xW[5] + 2.9744e+06 xW[6]
   + 2.9744e+06 xW[7] + 2.9744e+06 xW[8] + 2.9744e+06 xW[9]
   + 2.9744e+06 xW[10] + 2.9744e+06 xW[11] + 2.9744e+06 xW[12]
   + 2.9744e+06 xW[13] + 2.9744e+06 xW[14] + 2.9744e+06 xW[15]
   + 2.9744e+06 xW[16] + 2.9744e+06 xW[17] + 2.9744e+06 xW[18]
   + 2.9744e+06 xW[19] + 2.9744e+06 xW[20] + 2.9744e+06 xW[21]
   + 2.9744e+06 xW[22] + 2.9744e+06 xW[23] + gamma
Subject To
 Fix_xW_others[0]: xW[0] = 0
 Fix_xW_others[1]: xW[1] = 0
 Fix_xW_others[2]: xW[2] = 0
 Fix_xW_others[3]: xW[3] = 0
 Fix_xW_others[4]: xW[4] = 0
 Fix_xW_others[5]: xW[5] = 0
 Fix_xW_others[6]: xW[6] = 0
 Fix_xW_others[7]: xW[7] = 0
 Fix_xW_others[8]: xW[8] = 0
 Fix_xW_others[9]: xW[9] = 0
 Fix_xW_others[10]: xW[10] = 0
 Fix_xW_others[11]: xW[11] = 0
 Fix_xW_others[12]: xW[12] = 0
 Fix_xW_others[14]: xW[14] = 0
 Fix_xW_others[15]: xW[15] = 0
 Fix_xW_others[16]: xW[16] = 0
 Fix_xW_others[17]: xW[17] = 0
 Fix_xW_others[18]: xW[18] = 0
 Fix_xW_others[19]: xW[19] = 0
 Fix_xW_others[20]: xW[20] = 0
 Fix_xW_others[21]: xW[21] = 0
 Fix_xW_others[22]: xW[22] = 0
 Fix_xW_others[23]: xW[23] = 0
 Total_Capacity: 2.9744e+06 xW[0] + 2.9744e+06 xW[1] + 2.9744e+06 xW[2]
   + 2.9744e+06 xW[3] + 2.9744e+06 xW[4] + 2.9744e+06 xW[5]
   + 2.9744e+06 xW[6] + 2.9744e+06 xW[7] + 2.9744e+06 xW[8]
   + 2.9744e+06 xW[9] + 2.9744e+06 xW[10] + 2.9744e+06 xW[11]
   + 2.9744e+06 xW[12] + 2.9744e+06 xW[13] + 2.9744e+06 xW[14]
   + 2.9744e+06 xW[15] + 2.9744e+06 xW[16] + 2.9744e+06 xW[17]
   + 2.9744e+06 xW[18] + 2.9744e+06 xW[19] + 2.9744e+06 xW[20]
   + 2.9744e+06 xW[21] + 2.9744e+06 xW[22] + 2.9744e+06 xW[23] <= 1.6e+10
 GammaLowerBound: gamma >= 0
 GammaUpperBound: gamma <= 1e+06
 BendersCut_0: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_1: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_2: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_3: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_4: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_5: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_6: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_7: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_8: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
 BendersCut_9: 1.1245634183333328e+07 xW[13] + gamma
   >= -1.6340599741359907e+10
Bounds
 xW[0] <= 150
 xW[1] <= 150
 xW[2] <= 150
 xW[3] <= 150
 xW[4] <= 150
 xW[5] <= 150
 xW[6] <= 150
 xW[7] <= 150
 xW[8] <= 150
 xW[9] <= 150
 xW[10] <= 150
 xW[11] <= 150
 xW[12] <= 150
 xW[13] <= 150
 xW[14] <= 150
 xW[15] <= 150
 xW[16] <= 150
 xW[17] <= 150
 xW[18] <= 150
 xW[19] <= 150
 xW[20] <= 150
 xW[21] <= 150
 xW[22] <= 150
 xW[23] <= 150
 gamma free
End
