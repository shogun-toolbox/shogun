%newobject shogun::distance(std::string name);
%newobject shogun::evaluation(const std::string& name);
%newobject shogun::kernel(const std::string& name);
%newobject shogun::machine(const std::string& name);
%newobject shogun::multiclass_strategy(const std::string& name);
%newobject shogun::ecoc_encoder(const std::string& name);
%newobject shogun::ecoc_decoder(const std::string& name);
%newobject shogun::transformer(const std::string& name);
%newobject shogun::layer(const std::string& name);
%newobject shogun::splitting_strategy(const std::string& name);
%newobject shogun::machine_evaluation(const std::string& name);
%newobject shogun::svm(const std::string& name);
%newobject shogun::features;
%newobject shogun::gp_likelihood(const std::string& name);
%newobject shogun::gp_mean(const std::string& name);
%newobject shogun::differentiable(const std::string& name);
%newobject shogun::gp_inference(const std::string& name);
%newobject shogun::loss(const std::string& name);
%newobject shogun::string_features(CFile*, EAlphabet alpha = DNA, EPrimitiveType primitive_type = PT_CHAR);
%newobject shogun::transformer(const std::string&);
%newobject shogun::csv_file(std::string fname, char rw);
%newobject shogun::libsvm_file(std::string fname, char rw);
%newobject shogun::pipeline;
%newobject shogun::labels;

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
