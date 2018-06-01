%{
    #include <shogun/util/factory.h>
%}
%include <shogun/util/factory.h>

%template(features) shogun::features<float64_t>;

/* These functions return a new Object */
%newobject distance(const std::string&);
%newobject evaluation(const std::string&);
%newobject kernel(const std::string&);
%newobject kernel(SGMatrix<float64_t>);
%newobject machine(const std::string&);
%newobject multiclass_strategy(const std::string&);
%newobject ecoc_encoder(const std::string&);
%newobject ecoc_decoder(const std::string&);
%newobject features(SGMatrix<float64_t>);
%newobject features(CFile*, EPrimitiveType);
%newobject features_subset(CFeatures*, SGVector<index_t>, EPrimitiveType);
%newobject labels(CFile*);
%newobject csv_file(std::string, char);
