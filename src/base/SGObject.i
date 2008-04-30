%{
 #include "base/SGObject.h" 
%}

#ifndef HAVE_R
%feature("ref")   CSGObject "SG_REF($this);"
%feature("unref") CSGObject "SG_UNREF($this);"
#endif

%rename(SGObject) CSGObject;

%include "base/SGObject.h" 
