%{
 #include "evaluation/PerformanceMeasures.h"
%}

%rename(PerformanceMeasures) CPerformanceMeasures;

#ifdef HAVE_PYTHON
%feature("autodoc", "get_ROC(self) -> numpy array of ROC points (2 floats)") get_ROC;
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** result, INT* dim, INT* num)};
%feature("autodoc", "get_accROC(self) -> numpy array of accuracy values (float), aligned to ROC") get_accROC;
%feature("autodoc", "get_errROC(self) -> numpy array of error rate values (float), aligned to ROC") get_errROC;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** result, INT* num)};
#endif

%include "evaluation/PerformanceMeasures.h"
