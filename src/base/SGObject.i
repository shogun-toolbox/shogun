%{
 #include "base/SGObject.h" 
%}

%feature("ref")   CSGObject "SG_REF($this);"
%feature("unref") CSGObject "SG_UNREF($this);"

%rename(SGObject) CSGObject;

%include "base/SGObject.h" 
