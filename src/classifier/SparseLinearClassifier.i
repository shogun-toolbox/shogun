%{
 #include "classifier/SparseLinearClassifier.h"
%}
%rename(SparseLinearClassifier) CSparseLinearClassifier;

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%feature("autodoc", "get_w(self) -> [] of float") get_w;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst_w, INT* dst_dims)};
#endif

%include "classifier/SparseLinearClassifier.h"
%include "classifier/svm/SVMLin.i"
%include "classifier/svm/LibLinear.i"
%include "classifier/svm/SubGradientSVM.i"
%include "classifier/SubGradientLPM.i"
%include "classifier/svm/SVMOcas.i"
%include "classifier/LPM.i"
%include "classifier/LPBoost.i"
