%{
 #include "classifier/LinearClassifier.h" 
%}

%newobject classify;
%rename(LinearClassifier) CLinearClassifier;

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
#endif

%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dst_w, int32_t* dst_dims)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* src_w, int32_t src_w_dim)};

%include "classifier/LinearClassifier.h" 
%include "classifier/Perceptron.i"
%include "classifier/LDA.i"
%include "classifier/svm/SVMLin.i"
%include "classifier/svm/LibLinear.i"
%include "classifier/svm/SubGradientSVM.i"
%include "classifier/svm/SVMSGD.i"
%include "classifier/SubGradientLPM.i"
%include "classifier/svm/SVMOcas.i"
%include "classifier/LPM.i"
%include "classifier/LPBoost.i"
