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
