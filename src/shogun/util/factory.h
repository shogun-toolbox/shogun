/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */
#ifndef FACTORY_H_
#define FACTORY_H_

#include <shogun/base/class_list.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/CSVFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/DenseLabels.h>

namespace shogun
{

	CKernel* kernel(const std::string& name);
	CMachine* machine(const std::string& name);

#define BASE_CLASS_FACTORY(T, factory_name)                                    \
	T* factory_name(const std::string& name)                                   \
	{                                                                          \
		return create_object<T>(name.c_str());                                 \
	}

	BASE_CLASS_FACTORY(CKernel, kernel)
	BASE_CLASS_FACTORY(CMachine, machine)

	template <class T>
	CFeatures* features(SGMatrix<T> mat)
	{
		CFeatures* features = new CDenseFeatures<T>(mat);
		SG_REF(features);
		return features;
	}

	CFeatures* features(CFile* file, EPrimitiveType primitive_type)
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

	CFile* csv_file(std::string fname, char rw)
	{
		CFile* result = new CCSVFile(fname.c_str(), rw);
		SG_REF(result);
		return result;
	}
}
#endif // FACTORY_H_
