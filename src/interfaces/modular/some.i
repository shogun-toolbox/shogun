%{
    #include <shogun/base/some.h>
    #include <shogun/base/SGObject.h>
    #include <shogun/base/Parallel.h>
    #include <shogun/io/SGIO.h>
%}

%rename (ptr) operator CSGObject*;
%rename (ptrSGIO) operator SGIO*;
%rename (ptrParallel) operator Parallel*;
%include <shogun/base/some.h>

%template (SomeSGObject) shogun::Some<CSGObject>;
%template (SomeSGIO) shogun::Some<SGIO>;
%template (SomeParallel) shogun::Some<Parallel>;
