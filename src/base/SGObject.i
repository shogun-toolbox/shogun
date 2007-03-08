%{
 #include "base/SGObject.h" 
%}

%feature("ref")   CSGObject "$this->ref();"
%feature("unref") CSGObject "$this->unref();"

%rename(SGObject) CSGObject;

%include "base/SGObject.h" 
