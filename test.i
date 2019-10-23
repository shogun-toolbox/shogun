%module test

%{
  #include "Base.h"
  #include "Derived.h"
  #include "Interface.h"
%}

%include "Base.h"
%template(testClass) Base<Derived<float>>;
%include "Derived.h"
%template(FloatDerived) Derived<float>;


//%include "Interface.h"

