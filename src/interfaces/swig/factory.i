%{
    #include <shogun/util/factory.h>
%}
%include <shogun/util/factory.h>

%template(features) shogun::features<float64_t>;

%newobject shogun::string_features(CFile*, EAlphabet alpha = DNA, EPrimitiveType primitive_type = PT_CHAR);
