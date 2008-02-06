%{
 #include "evaluation/PerformanceMeasures.h"
%}

%rename(PerformanceMeasures) CPerformanceMeasures;

#ifdef HAVE_PYTHON
%feature("autodoc", "compute_ROC(self) -> numpy array of 2 floats") compute_ROC;
%apply (DREAL** ARGOUT2, INT* DIM1, INT* DIM2) {(DREAL** result, INT* dim, INT* num)};
#endif

%include "evaluation/PerformanceMeasures.h"
