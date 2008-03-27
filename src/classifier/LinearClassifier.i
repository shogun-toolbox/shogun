%{
 #include "classifier/LinearClassifier.h" 
%}

%newobject classify;
%rename(LinearClassifier) CLinearClassifier;

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%feature("autodoc", "get_w(self) -> [] of float") get_w;
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst_w, INT* dst_dims)};
#endif

%include "classifier/LinearClassifier.h" 
%include "classifier/Perceptron.i"
%include "classifier/LDA.i"
