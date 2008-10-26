%{
 #include "classifier/LinearClassifier.h" 
%}

%newobject classify;
%rename(LinearClassifier) CLinearClassifier;

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
#endif

%apply (DREAL** ARGOUT1, int32_t* DIM1) {(DREAL** dst_w, int32_t* dst_dims)};
%apply (DREAL* IN_ARRAY1, int32_t DIM1) {(DREAL* src_w, int32_t src_w_dim)};

%include "classifier/LinearClassifier.h" 
%include "classifier/Perceptron.i"
%include "classifier/LDA.i"
