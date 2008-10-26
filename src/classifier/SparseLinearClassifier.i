%{
 #include "classifier/SparseLinearClassifier.h"
%}

%newobject classify;
%rename(SparseLinearClassifier) CSparseLinearClassifier;

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
#endif

%apply (DREAL** ARGOUT1, int32_t* DIM1) {(DREAL** dst_w, int32_t* dst_dims)};

%include "classifier/SparseLinearClassifier.h"
%include "classifier/svm/SVMLin.i"
%include "classifier/svm/LibLinear.i"
%include "classifier/svm/SubGradientSVM.i"
%include "classifier/svm/SVMSGD.i"
%include "classifier/SubGradientLPM.i"
%include "classifier/svm/SVMOcas.i"
%include "classifier/LPM.i"
%include "classifier/LPBoost.i"
