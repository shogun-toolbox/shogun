%{
 #include <shogun/classifier/LinearClassifier.h> 
%}

%newobject classify;
%rename(LinearClassifier) CLinearClassifier;

%include "swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
#endif

%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst_w, int32_t* dst_dims)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* src_w, int32_t src_w_dim)};

%include <shogun/classifier/LinearClassifier.h> 
%include "Perceptron.i"
%include "LDA.i"
%include "SVMLin.i"
%include "LibLinear.i"
%include "SubGradientSVM.i"
%include "SVMSGD.i"
%include "SubGradientLPM.i"
%include "SVMOcas.i"
%include "LPM.i"
%include "LPBoost.i"
