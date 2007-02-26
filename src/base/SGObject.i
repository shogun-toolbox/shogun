%{
 #include "base/SGObject.h" 
%}

%feature("ref")   CSGObject "$this->ref();"
%feature("unref") CSGObject "$this->unref();"

%rename(SGObject) CSGObject;

%include "lib/io.i" 
%include "base/SGObject.h" 
