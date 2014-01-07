#include <labels/LabelsFactory.h>

#include <labels/BinaryLabels.h>
#include <labels/LatentLabels.h>
#include <labels/MulticlassLabels.h>
#include <labels/RegressionLabels.h>
#include <labels/StructuredLabels.h>
#include <labels/MulticlassMultipleOutputLabels.h>

using namespace shogun;

CBinaryLabels* CLabelsFactory::to_binary(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	if (base_labels->get_label_type() == LT_BINARY)
		return static_cast<CBinaryLabels*>(base_labels);
	else
		SG_SERROR("base_labels must be of dynamic type CBinaryLabels")

	return NULL;
}

CLatentLabels* CLabelsFactory::to_latent(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	if (base_labels->get_label_type() == LT_LATENT)
		return static_cast<CLatentLabels*>(base_labels);
	else
		SG_SERROR("base_labels must be of dynamic type CLatentLabels\n")

	return NULL;
}

CMulticlassLabels* CLabelsFactory::to_multiclass(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	if (base_labels->get_label_type() == LT_MULTICLASS)
		return static_cast<CMulticlassLabels*>(base_labels);
	else
		SG_SERROR("base_labels must be of dynamic type CMulticlassLabels\n")

	return NULL;
}

CRegressionLabels* CLabelsFactory::to_regression(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	if (base_labels->get_label_type() == LT_REGRESSION)
		return static_cast<CRegressionLabels*>(base_labels);
	else
		SG_SERROR("base_labels must be of dynamic type CRegressionLabels")

	return NULL;
}

CStructuredLabels* CLabelsFactory::to_structured(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	if (base_labels->get_label_type() == LT_STRUCTURED)
		return static_cast<CStructuredLabels*>(base_labels);
	else
		SG_SERROR("base_labels must be of dynamic type CStructuredLabels\n")

	return NULL;
}


CMulticlassMultipleOutputLabels* CLabelsFactory::to_multiclass_multiple_output(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	if (base_labels->get_label_type() == LT_MULTICLASS_MULTIPLE_OUTPUT)
		return static_cast<CMulticlassMultipleOutputLabels*>(base_labels);
	else
		SG_SERROR("base_labels must be of dynamic type CMulticlassMultipleOutputLabels\n")

	return NULL;
}
