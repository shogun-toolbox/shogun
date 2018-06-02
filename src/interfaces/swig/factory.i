%{
    #include <shogun/util/factory.h>
%}
%include <shogun/util/factory.h>

%template(features) shogun::features<float64_t>;

%newobject shogun::transformer(const std::string&);
