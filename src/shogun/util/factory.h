/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Fernando Iglesias
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include <shogun/base/class_list.h>
#include <shogun/distance/Distance.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/io/CSVFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/machine/Machine.h>
#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>
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

	template <class T>
	CFeatures* features(SGMatrix<T> mat)
	{
		CFeatures* features = new CDenseFeatures<T>(mat);
		SG_REF(features);
		return features;
	}

	CFeatures* features(CFile* file, EPrimitiveType primitive_type = PT_FLOAT64)
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

	CLabels* labels(CFile* file)
	{
		REQUIRE(file, "No file provided.\n");
		CLabels* result = nullptr;

		if (dynamic_cast<CCSVFile*>(file))
		{
			CDenseLabels* result_ = new CDenseLabels();
			result_->load(file);
			result = result_;
		}
		else
			SG_SERROR("Cannot load labels from file %s.\n", file->get_name());

		SG_REF(result);
		return result;
	}

	CFile* csv_file(std::string fname, char rw = 'r')
	{
		CFile* result = new CCSVFile(fname.c_str(), rw);
		SG_REF(result);
		return result;
	}
}
#endif // FACTORY_H_
