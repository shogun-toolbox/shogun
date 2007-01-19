%{
 #include "classifier/LinearClassifier.h" 
%}
%rename(LinearClassifier) CLinearClassifier;

#ifdef HAVE_PYTHON
%include "lib/numpy.i"
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst_w, INT* dst_dims)};
#endif

%include "classifier/Classifier.i" 
%include "classifier/LinearClassifier.h" 
