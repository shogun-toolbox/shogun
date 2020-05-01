%{
    #include <shogun/util/factory.h>
%}
%include <shogun/util/factory.h>
%include <std_string.i>

%inline%{
namespace shogun{
	template <typename T>
	std::shared_ptr<Features> create_features(SGMatrix<T> mat)
	{
		return create<Features>(mat);
	}

	std::shared_ptr<Features> create_features(
	    std::shared_ptr<File> file, EPrimitiveType primitive_type = PT_FLOAT64)
	{
		return create<Features>(file, primitive_type);
	}

    template <
	    typename T, typename T2 = typename std::enable_if_t<
	                    std::is_floating_point<T>::value>>
	std::shared_ptr<Kernel> create_kernel(SGMatrix<T> kernel_matrix)
	{
		return details::kernel(kernel_matrix);
	}

	template <typename T>
	std::shared_ptr<Labels> create_labels(SGVector<T> labels)
	{
		return create<Labels>(labels);
	}

	std::shared_ptr<Labels> create_labels(std::shared_ptr<File> file)
	{
		return create<Labels>(file);
	}

	std::shared_ptr<File> read_csv(std::string fname, char rw = 'r')
	{
		return create<CSVFile>(fname, rw);
	}

	std::shared_ptr<File> read_libsvm(std::string fname, char rw = 'r')
	{
		return create<LibSVMFile>(fname, rw);
	}
	std::shared_ptr<PipelineBuilder> create_pipeline()
	{
		return details::pipeline();
	} 
	std::shared_ptr<Features> read_string_features(
	    std::shared_ptr<File> file, EAlphabet alphabet_type = DNA,
	    EPrimitiveType primitive_type = PT_CHAR)
	{
		return create<StringFeatures<char>>(file, alphabet_type, primitive_type);
	}

	std::shared_ptr<Features> create_features_subset(
	    std::shared_ptr<Features> base_features, SGVector<index_t> indices,
	    EPrimitiveType primitive_type = PT_FLOAT64)
	{
		return create<DenseSubsetFeatures<float64_t>>(base_features, indices, primitive_type);
	}

	std::shared_ptr<Features> create_string_features(
	    std::shared_ptr<Features> features, int32_t start, int32_t p_order,
	    int32_t gap, bool rev, EPrimitiveType primitive_type = PT_UINT16)
	{
		return create<StringFeatures<uint16_t>>(features, start, p_order, gap, rev, primitive_type);
	}   
}
%}
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
%template(create_features) shogun::create<shogun::Features, std::string>;
%template(create_machine) shogun::create<shogun::Machine, std::string>;
%template(create_structured_model) shogun::create<shogun::StructuredModel, std::string>;
%template(create_factor_type) shogun::create<shogun::FactorType, std::string>;
%template(create_gaussian_process) shogun::create<shogun::GaussianProcess, std::string>;
%template(create_labels) shogun::create_labels<float64_t>;