/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Fernando Iglesias
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include <shogun/base/class_list.h>
#include <shogun/classifier/svm/SVM.h>
#include <shogun/converter/Converter.h>
#include <shogun/distance/Distance.h>
#include <shogun/distributions/Distribution.h>
#include <shogun/ensemble/CombinationRule.h>
#include <shogun/evaluation/DifferentiableFunction.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/io/CSVFile.h>
#include <shogun/io/LibSVMFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/lib/observers/ParameterObserver.h>
#include <shogun/loss/LossFunction.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/Pipeline.h>
#include <shogun/machine/gp/Inference.h>
#include <shogun/machine/gp/LikelihoodModel.h>
#include <shogun/machine/gp/MeanFunction.h>
#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

	std::shared_ptr<Distance> distance(const std::string& name);
	std::shared_ptr<Evaluation> evaluation(const std::string& name);
	std::shared_ptr<Kernel> kernel(const std::string& name);
	std::shared_ptr<Machine> machine(const std::string& name);
	std::shared_ptr<MulticlassStrategy> multiclass_strategy(const std::string& name);
	std::shared_ptr<ECOCEncoder> ecoc_encoder(const std::string& name);
	std::shared_ptr<ECOCDecoder> ecoc_decoder(const std::string& name);
	std::shared_ptr<Transformer> transformer(const std::string& name);
	std::shared_ptr<NeuralLayer> layer(const std::string& name);
	std::shared_ptr<SplittingStrategy> splitting_strategy(const std::string& name);
	std::shared_ptr<MachineEvaluation> machine_evaluation(const std::string& name);
	std::shared_ptr<SVM> svm(const std::string& name);
	std::shared_ptr<Features> features(const std::string& name);
	std::shared_ptr<LikelihoodModel> gp_likelihood(const std::string& name);
	std::shared_ptr<MeanFunction> gp_mean(const std::string& name);
	std::shared_ptr<DifferentiableFunction> differentiable(const std::string& name);
	std::shared_ptr<Inference> gp_inference(const std::string& name);
	std::shared_ptr<LossFunction> loss(const std::string& name);
	std::shared_ptr<ParameterObserver> parameter_observer(const std::string& name);
	std::shared_ptr<Distribution> distribution(const std::string& name);
	std::shared_ptr<CombinationRule> combination_rule(const std::string& name);

#define BASE_CLASS_FACTORY(T, factory_name)                                    \
	std::shared_ptr<T> factory_name(const std::string& name)                   \
	{                                                                          \
		return create_object<T>(name.c_str());                                 \
	}                                                                          \
	std::shared_ptr<T> as_## factory_name(std::shared_ptr<SGObject> obj)      \
	{                                                                          \
		return obj->as<T>();                             					   \
	}
	BASE_CLASS_FACTORY(Evaluation, evaluation)
	BASE_CLASS_FACTORY(Distance, distance)
	BASE_CLASS_FACTORY(Kernel, kernel)
	BASE_CLASS_FACTORY(Machine, machine)
	BASE_CLASS_FACTORY(MulticlassStrategy, multiclass_strategy)
	BASE_CLASS_FACTORY(ECOCEncoder, ecoc_encoder)
	BASE_CLASS_FACTORY(ECOCDecoder, ecoc_decoder)
	BASE_CLASS_FACTORY(Transformer, transformer)
	BASE_CLASS_FACTORY(NeuralLayer, layer)
	BASE_CLASS_FACTORY(SplittingStrategy, splitting_strategy)
	BASE_CLASS_FACTORY(MachineEvaluation, machine_evaluation)
	BASE_CLASS_FACTORY(SVM, svm)
	BASE_CLASS_FACTORY(Features, features)
	BASE_CLASS_FACTORY(LikelihoodModel, gp_likelihood)
	BASE_CLASS_FACTORY(MeanFunction, gp_mean)
	BASE_CLASS_FACTORY(Inference, gp_inference)
	BASE_CLASS_FACTORY(DifferentiableFunction, differentiable)
	BASE_CLASS_FACTORY(LossFunction, loss)
	BASE_CLASS_FACTORY(ParameterObserver, parameter_observer)
	BASE_CLASS_FACTORY(EvaluationResult, evaluation_result)
	BASE_CLASS_FACTORY(Distribution, distribution)
	BASE_CLASS_FACTORY(CombinationRule, combination_rule)

	template <class T>
	std::shared_ptr<Features> features(SGMatrix<T> mat)
	{
		return std::make_shared<DenseFeatures<T>>(mat);
	}

	std::shared_ptr<Features> features(std::shared_ptr<File> file, EPrimitiveType primitive_type = PT_FLOAT64)
	{
		require(file, "No file provided.");
		std::shared_ptr<Features> result = nullptr;

		switch (primitive_type)
		{
		case PT_FLOAT64:
			result = std::make_shared<DenseFeatures<float64_t>>();
			break;
		case PT_FLOAT32:
			result = std::make_shared<DenseFeatures<float32_t>>();
			break;
		case PT_FLOATMAX:
			result = std::make_shared<DenseFeatures<floatmax_t>>();
			break;
		case PT_UINT8:
			result = std::make_shared<DenseFeatures<uint8_t>>();
			break;
		case PT_UINT16:
			result = std::make_shared<DenseFeatures<uint16_t>>();
			break;
		default:
			not_implemented(SOURCE_LOCATION);
		}
		result->load(file);
		return result;
	}

	std::shared_ptr<Features> string_features(
	    std::shared_ptr<File> file, EAlphabet alphabet_type = DNA,
	    EPrimitiveType primitive_type = PT_CHAR)
	{
		require(file, "No file provided.");

		switch (primitive_type)
		{
		case PT_CHAR:
		{
			return std::make_shared<StringFeatures<char>>(file, alphabet_type);
		}
		default:
			not_implemented(SOURCE_LOCATION);
		}

		return nullptr;
	}

	/** Create embedded string features from string char features.
	 * The new features has the same alphabet as the original features. Data of
	 * the new features is obtained by calling CStringFeatures::obtain_from_char
	 * with the given features and other arguments of this factory method.
	 *
	 * @param features StringCharFeatures
	 * @param start start
	 * @param p_order order
	 * @param gap gap
	 * @param rev reverse
	 * @param primitive_type primitive type of the string features
	 * @return new instance of string features
	 */
	std::shared_ptr<Features> string_features(
	    std::shared_ptr<Features> features, int32_t start, int32_t p_order, int32_t gap,
	    bool rev, EPrimitiveType primitive_type = PT_UINT16)
	{

		require<std::invalid_argument>(features, "No features provided.");
		require<std::invalid_argument>(
		    features->get_feature_class() == C_STRING &&
		        features->get_feature_type() == F_CHAR,
		    "Given features must be char-based StringFeatures, "
		    "provided ({}) have feature class ({}), feature type "
		    "({}) and class name.",
		    features->get_name(), features->get_feature_class(),
		    features->get_feature_type());

		auto string_features = std::dynamic_pointer_cast<StringFeatures<char>>(features);

		switch (primitive_type)
		{
		case PT_UINT16:
		{
			auto result =
			    std::make_shared<StringFeatures<uint16_t>>(string_features->get_alphabet());
			bool success = result->obtain_from_char(
			    string_features, start, p_order, gap, rev);
			require(success, "Failed to obtain from string char features.");
			return result;
		}
		default:
			not_implemented(SOURCE_LOCATION);
		}

		return nullptr;
	}

	/** Factory for CDenseSubsetFeatures.
	 * TODO: Should be removed once the concept of feature views has arrived
	 */
	std::shared_ptr<Features> features_subset(std::shared_ptr<Features> base_features, SGVector<index_t> indices,
			EPrimitiveType primitive_type = PT_FLOAT64)
	{
		require(base_features, "No base features provided.");

		switch (primitive_type)
		{
		case PT_FLOAT64:
			return std::make_shared<DenseSubsetFeatures<float64_t>>(
				std::dynamic_pointer_cast<DenseFeatures<float64_t>>(base_features), indices);
			break;
		default:
			not_implemented(SOURCE_LOCATION);
		}

		return nullptr;
	}

	template <typename T, typename T2 = typename std::enable_if_t<
	                          std::is_floating_point<T>::value>>
	std::shared_ptr<Kernel> kernel(SGMatrix<T> kernel_matrix)
	{
		return std::make_shared<CustomKernel>(kernel_matrix);
	}

#ifndef SWIG // SWIG should skip this part
	template <typename LT,
	          std::enable_if_t<
	              std::is_base_of_v<DenseLabels, typename std::remove_pointer_t<LT>>,
	              LT>* = nullptr>
	void try_labels(std::shared_ptr<DenseLabels>& labels, const SGVector<float64_t>& data)
	{
		if (!labels)
		{
			auto l = std::make_shared<LT>();
			l->set_labels(data);
			if (l->is_valid())
				labels = l;
		}
	}
#endif // SWIG

	std::shared_ptr<Labels> labels(std::shared_ptr<File> file)
	{
		require(file, "No file provided.");

		// load label data into memory via any dense label specialization
		auto loaded = std::make_shared<RegressionLabels>();
		loaded->load(file);
		auto labels = loaded->get_labels();

		std::shared_ptr<DenseLabels> result = nullptr;
		auto csv_file = std::dynamic_pointer_cast<CSVFile>(file);
		require(
		    file,
		    "Cannot load labels from {}(\"{}\").", file->get_name(),
		    file->get_filename());
		// try to interpret as all dense label types, from most restrictive to
		// least restrictive
		try_labels<BinaryLabels>(result, labels);
		try_labels<MulticlassLabels>(result, labels);
		try_labels<RegressionLabels>(result, labels);
		require(
		    result,
		    "Cannot load labels from {}(\"{}\") as any of dense labels.",
		    file->get_name(), file->get_filename());
		io::info(
		    "Loaded labels from {}(\"{}\") as {}", file->get_name(),
		    file->get_filename(), result->get_name());

		return result;
	}

	template <class T>
	std::shared_ptr<Labels> labels(SGVector<T> labels)
	{
		std::shared_ptr<DenseLabels> result = nullptr;
		// try to interpret as all dense label types, from most restrictive to
		// least restrictive
		try_labels<BinaryLabels>(result, labels);
		try_labels<MulticlassLabels>(result, labels);
		try_labels<RegressionLabels>(result, labels);
		require(
		    result, "Cannot interpret given labels as any of dense labels.");
		io::info("Interpreted labels as {}", result->get_name());
		return result;
	}

	std::shared_ptr<File> csv_file(std::string fname, char rw = 'r')
	{
		return std::make_shared<CSVFile>(fname.c_str(), rw);
	}

	std::shared_ptr<File> libsvm_file(std::string fname, char rw = 'r')
	{
		return std::make_shared<LibSVMFile>(fname.c_str(), rw);
	}

	/** Create a pipeline builder.
	 * See also PipelineBuilder and CPipeline.
	 * @return new instance of PipelineBuilder
	 */
	std::shared_ptr<PipelineBuilder> pipeline()
	{
		return std::make_shared<PipelineBuilder>();
	}
} // namespace shogun
#endif // FACTORY_H_
