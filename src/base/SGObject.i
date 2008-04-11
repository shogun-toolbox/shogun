%{
 #include "base/SGObject.h" 
%}

#ifdef HAVE_PYTHON
%feature("ref")   CSGObject "SG_REF($this);"
%feature("unref") CSGObject "SG_UNREF($this);"
#endif

%rename(SGObject) CSGObject;

%include "base/SGObject.h" 
