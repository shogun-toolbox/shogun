%{
 #include "features/Labels.h" 
%}

%include "lib/numpy.i"
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* labels, INT len)};

%include "features/Labels.h" 

/* 
 * This stuff is omitted because of the delete
 * problems when interfacing with python
 * ( blah.thisown and friends ).
 *
 */

%ignore CLabels::CLabels(INT num_labels);
%ignore CLabels::CLabels(CHAR* fname);

%pythoncode %{
  class Labels(CLabels):
     def __init__(self,p1): 
        CLabels.__init__(self)
        self.set_labels(p1)
%}
