%{
 #include "classifier/LinearClassifier.h" 
%}

%newobject classify;
%rename(LinearClassifier) CLinearClassifier;

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
#endif

%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst_w, INT* dst_dims)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* src_w, INT src_w_dim)};

%include "classifier/LinearClassifier.h" 
%include "classifier/Perceptron.i"
%include "classifier/LDA.i"
