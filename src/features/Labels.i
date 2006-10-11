%{
 #include "features/Labels.h" 
%}

%include "features/Labels.h" 

%include "carrays.i"
%array_class(CLabels,labelArray)

/* 
 * This stuff is omitted because of the delete
 * problems when interfacing with python
 * ( blah.thisown and friends ).
 *
 */

%ignore CLabels::CLabels(INT num_labels);
%ignore CLabels::CLabels(CHAR* fname);
