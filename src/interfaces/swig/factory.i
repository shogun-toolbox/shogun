%{
    #include <shogun/util/factory.h>
%}
%include <shogun/util/factory.h>
%include <std_string.i>

%template(create_features) shogun::create_features<float64_t>;

#ifndef SWIGJAVA // FIXME: Java only uses DoubleMatrix atm, remove guard once that is resolved

%template(create_features) shogun::create_features<uint16_t>;
%template(create_features) shogun::create_features<int64_t>;

#endif //SWIGJAVA

%template(create_svm) shogun::create<shogun::SVM,std::string>;
%template(create_evaluation) shogun::create<shogun::Evaluation, std::string>;
%template(create_multiclass_strategy) shogun::create<shogun::MulticlassStrategy, std::string>;
%template(create_ecoc_encoder) shogun::create<shogun::ECOCEncoder, std::string>;
%template(create_ecoc_decoder) shogun::create<shogun::ECOCDecoder, std::string>;
%template(create_transformer) shogun::create<shogun::Transformer, std::string>;
%template(create_layer) shogun::create<shogun::NeuralLayer, std::string>;
%template(create_splitting_strategy) shogun::create<shogun::SplittingStrategy, std::string>;
%template(create_machine_evaluation) shogun::create<shogun::MachineEvaluation, std::string>;
%template(create_gp_likelihood) shogun::create<shogun::LikelihoodModel, std::string>;
%template(create_gp_mean) shogun::create<shogun::MeanFunction, std::string>;
%template(create_gp_inference) shogun::create<shogun::Inference, std::string>;
%template(create_differentiable) shogun::create<shogun::DifferentiableFunction, std::string>;
%template(create_loss) shogun::create<shogun::LossFunction, std::string>;
%template(create_parameter_observer) shogun::create<shogun::ParameterObserver, std::string>;
%template(create_evaluation_result) shogun::create<shogun::EvaluationResult, std::string>;
%template(create_distribution) shogun::create<shogun::Distribution, std::string>;
%template(create_combination_rule) shogun::create<shogun::CombinationRule, std::string>;
%template(create_distance) shogun::create<shogun::Distance, std::string>;
%template(create_kernel) shogun::create<shogun::Kernel, std::string>;
%template(create_machine) shogun::create<shogun::Machine, std::string>;
%template(create_labels) shogun::create_labels<float64_t>;
