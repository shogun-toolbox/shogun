%{
    #include <shogun/util/factory.h>
%}
%include <shogun/util/factory.h>

%template(features) shogun::features<float64_t>;
#ifndef SWIGJAVA // FIXME: Java only uses DoubleMatrix atm, remove guard once that is resolved
%template(features) shogun::features<uint16_t>;
%template(features) shogun::features<int64_t>;
#endif //SWIGJAVA

%template(labels) shogun::labels<float64_t>;

%newobject shogun::features(CFile*, EPrimitiveType primitive_type);
%newobject shogun::string_features(CFile*, EAlphabet alpha = DNA, EPrimitiveType primitive_type = PT_CHAR);
%newobject shogun::transformer(const std::string&);
%newobject shogun::csv_file(std::string fname, char rw);
%newobject shogun::libsvm_file(std::string fname, char rw);
