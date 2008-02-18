%{
 #include "evaluation/PerformanceMeasures.h"
%}


#ifdef HAVE_PYTHON
%feature("autodoc", "get_ROC(self) -> numpy array of Receiver Operating Curve points (2 floats)") get_ROC;
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** result, INT* dim, INT* num)};
%feature("autodoc", "get_accuracyROC(self) -> numpy array of accuracy values (float), aligned to ROC") get_accuracyROC;
%feature("autodoc", "get_errorROC(self) -> numpy array of error rate values (float), aligned to ROC") get_errorROC;
%feature("autodoc", "get_PRC(self) -> numpy array of Precision Recall Curve points (2 floats)") get_PRC;
%feature("autodoc", "get_fmeasurePRC(self) -> numpy array of F-measure values %(float), aligned to PRC") get_fmeasurePRC;
%feature("autodoc", "get_DET(self) -> numpy array of Detection Error Tradeoff points (2 floats)") get_DET;
%feature("autodoc", "get_CC(self) -> numpy array of Cross Correlation coefficients (float)") get_CC;
%feature("autodoc", "get_WRAcc(self) -> numpy array of Weighted Relative Accuracy values (float)") get_WRAcc;
%feature("autodoc", "get_BAL(self) -> numpy array of Balanced Error values (float)") get_BAL;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** result, INT* num)};
#endif

%rename(PerformanceMeasures) CPerformanceMeasures;

%include "evaluation/PerformanceMeasures.h"
