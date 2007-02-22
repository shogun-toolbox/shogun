%{
 #include "classifier/SparseLinearClassifier.h" 
%}
%rename(SparseLinearClassifier) CSparseLinearClassifier;

#ifdef HAVE_PYTHON
%include "lib/numpy.i"
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** dst_w, INT* dst_dims)};
#endif

%include "classifier/SparseLinearClassifier.h" 
