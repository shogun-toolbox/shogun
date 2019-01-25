/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Fernando Iglesias
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include <shogun/base/class_list.h>
#include <shogun/converter/Converter.h>
#include <shogun/distance/Distance.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/io/CSVFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/Pipeline.h>
#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

	CDistance* distance(const std::string& name);
	CEvaluation* evaluation(const std::string& name);
	CKernel* kernel(const std::string& name);
	CMachine* machine(const std::string& name);
	CMulticlassStrategy* multiclass_strategy(const std::string& name);
	CECOCEncoder* ecoc_encoder(const std::string& name);
	CECOCDecoder* ecoc_decoder(const std::string& name);
	CTransformer* transformer(const std::string& name);
	CNeuralLayer* layer(const std::string& name);
	CSplittingStrategy* splitting_strategy(const std::string& name);

#define BASE_CLASS_FACTORY(T, factory_name)                                    \
	T* factory_name(const std::string& name)                                   \
	{                                                                          \
		return create_object<T>(name.c_str());                                 \
	}                                                                          \
	T* factory_name(CSGObject* obj)                                            \
	{                                                                          \
		return obj->as<T>();                                                   \
	}

	BASE_CLASS_FACTORY(CEvaluation, evaluation)
	BASE_CLASS_FACTORY(CDistance, distance)
	BASE_CLASS_FACTORY(CKernel, kernel)
	BASE_CLASS_FACTORY(CMachine, machine)
	BASE_CLASS_FACTORY(CMulticlassStrategy, multiclass_strategy)
	BASE_CLASS_FACTORY(CECOCEncoder, ecoc_encoder)
	BASE_CLASS_FACTORY(CECOCDecoder, ecoc_decoder)
	BASE_CLASS_FACTORY(CTransformer, transformer)
	BASE_CLASS_FACTORY(CNeuralLayer, layer)
	BASE_CLASS_FACTORY(CSplittingStrategy, splitting_strategy)

	template <class T>
	CFeatures* features(SGMatrix<T> mat)
	{
		CFeatures* features = new CDenseFeatures<T>(mat);
		SG_REF(features);
		return features;
	}

	CFeatures* features(CFile* file, machine_int_t primitive_type = PT_FLOAT64)
	{
		REQUIRE(file, "No file provided.\n");
		CFeatures* result = nullptr;

		if (dynamic_cast<CCSVFile*>(file))
		{
			switch (primitive_type)
			{
			case PT_FLOAT64:
				result = new CDenseFeatures<float64_t>();
				break;
			case PT_FLOAT32:
				result = new CDenseFeatures<float32_t>();
				break;
			case PT_FLOATMAX:
				result = new CDenseFeatures<floatmax_t>();
				break;
			default:
				SG_SNOTIMPLEMENTED
			}
			result->load(file);
		}
		else
			SG_SERROR("Cannot load features from %s.\n", file->get_name());

		SG_REF(result);
		return result;
	}

	CFeatures* string_features(
	    CFile* file, EAlphabet alpha = DNA,
	    EPrimitiveType primitive_type = PT_CHAR)
	{
		REQUIRE(file, "No file provided.\n");
		CFeatures* result = nullptr;

		switch (primitive_type)
		{
		case PT_CHAR:
			result = new CStringFeatures<char>(file, alpha);
			break;
		default:
			SG_SNOTIMPLEMENTED
		}

		SG_REF(result);
		return result;
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
	CFeatures* string_features(
	    CFeatures* features, int32_t start, int32_t p_order, int32_t gap,
	    bool rev, EPrimitiveType primitive_type)
	{

		REQUIRE_E(features, std::invalid_argument, "No features provided.\n");
		REQUIRE_E(
		    features->get_feature_class() == C_STRING &&
		        features->get_feature_type() == F_CHAR,
		    std::invalid_argument, "Only StringCharFeatures are supported, "
		                           "provided feature class (%d), feature type "
		                           "(%d).\n",
		    features->get_feature_class(), features->get_feature_type());

		auto string_features = features->as<CStringFeatures<char>>();

		switch (primitive_type)
		{
		case PT_UINT16:
		{
			auto result =
			    new CStringFeatures<uint16_t>(string_features->get_alphabet());
			bool success = result->obtain_from_char(
			    string_features, start, p_order, gap, rev);
			REQUIRE(success, "Failed to obtain from string char features.\n");
			SG_REF(result);
			return result;
		}
		default:
			SG_SNOTIMPLEMENTED
		}

		return nullptr;
	}

	/** Factory for CDenseSubsetFeatures.
	 * TODO: Should be removed once the concept of feature views has arrived
	 */
	CFeatures* features_subset(CFeatures* base_features, SGVector<index_t> indices,
			EPrimitiveType primitive_type = PT_FLOAT64)
	{
		CFeatures* result = nullptr;
		REQUIRE(base_features, "No base features provided.\n");

		switch (primitive_type)
		{
		case PT_FLOAT64:
			result = new CDenseSubsetFeatures<float64_t>(base_features->as<CDenseFeatures<float64_t>>(), indices);
			break;
		default:
			SG_SNOTIMPLEMENTED
		}

		SG_REF(result);
		return result;
	}

	template <typename T, typename T2 = typename std::enable_if_t<
	                          std::is_floating_point<T>::value>>
	CKernel* kernel(SGMatrix<T> kernel_matrix)
	{
		CKernel* result = new CCustomKernel(kernel_matrix);
		SG_REF(result);
		return result;
	}

#ifndef SWIG // SWIG should skip this part
	template <typename LT,
	          std::enable_if_t<
	              std::is_base_of<CDenseLabels, typename std::remove_pointer<
	                                                LT>::type>::value,
	              LT>* = nullptr>
	void try_labels(CDenseLabels*& labels, const SGVector<float64_t>& data)
	{
		if (!labels)
		{
			labels = new LT();
			labels->set_labels(data);
			if (!labels->is_valid())
				SG_UNREF(labels);
		}
	}
#endif // SWIG

	CLabels* labels(CFile* file)
	{
		REQUIRE(file, "No file provided.\n");

		// load label data into memory via any dense label specialization
		CDenseLabels* loaded = new CRegressionLabels();
		loaded->load(file);
		auto labels = loaded->get_labels();
		SG_UNREF(loaded);

		CDenseLabels* result = nullptr;

		REQUIRE(
		    dynamic_cast<CCSVFile*>(file),
		    "Cannot load labels from %s(\"%s\").\n", file->get_name(),
		    file->get_filename());

		// try to interpret as all dense label types, from most restrictive to
		// least restrictive
		try_labels<CBinaryLabels>(result, labels);
		try_labels<CMulticlassLabels>(result, labels);
		try_labels<CRegressionLabels>(result, labels);
		REQUIRE(
		    result,
		    "Cannot load labels from %s(\"%s\") as any of dense labels.\n",
		    file->get_name(), file->get_filename());
		SG_SINFO(
		    "Loaded labels from %s(\"%s\") as %s\n", file->get_name(),
		    file->get_filename(), result->get_name())

		SG_REF(result);
		return result;
	}

	template <class T>
	CLabels* labels(SGVector<T> labels)
	{
		CDenseLabels* result = nullptr;
		// try to interpret as all dense label types, from most restrictive to
		// least restrictive
		try_labels<CBinaryLabels>(result, labels);
		try_labels<CMulticlassLabels>(result, labels);
		try_labels<CRegressionLabels>(result, labels);
		REQUIRE(
		    result, "Cannot interpret given labels as any of dense labels.\n");
		SG_SINFO("Interpreted labels as %s\n", result->get_name())
		return result;
	}

	CFile* csv_file(std::string fname, char rw = 'r')
	{
		CFile* result = new CCSVFile(fname.c_str(), rw);
		SG_REF(result);
		return result;
	}

	/** Create a pipeline builder.
	 * See also CPipelineBuilder and CPipeline.
	 * @return new instance of CPipelineBuilder
	 */
	CPipelineBuilder* pipeline()
	{
		auto result = new CPipelineBuilder();
		SG_REF(result);
		return result;
	}

	CMachineEvaluation* machine_evaluation(const std::string& name, CMachine* machine)
	{
		auto obj = create_object<CMachineEvaluation>(name.c_str());
		obj->put("machine", machine);
		return obj;
	}

	CMachineEvaluation* machine_evaluation(CSGObject* obj)
	{
		return obj->as<CMachineEvaluation>();
	}
}
#endif // FACTORY_H_
