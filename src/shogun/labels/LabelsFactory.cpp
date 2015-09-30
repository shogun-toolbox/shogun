#include <shogun/labels/LabelsFactory.h>

#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/LatentLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/labels/MultilabelLabels.h>
#include <shogun/structure/MulticlassSOLabels.h>

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


CMultilabelLabels* CLabelsFactory::to_multilabel_output(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	if (base_labels->get_label_type() == LT_SPARSE_MULTILABEL)
		return static_cast<CMultilabelLabels*>(base_labels);
	else
		SG_SERROR("base_labels must be of dynamic type CMultilabelLabels\n")

	return NULL;
}

CMulticlassSOLabels* CLabelsFactory::to_multiclass_structured(CLabels* base_labels)
{
	ASSERT(base_labels != NULL)
	CMulticlassSOLabels* labels = dynamic_cast<CMulticlassSOLabels*>(base_labels);
	if (labels == NULL)
		SG_SERROR("base_labels must be of dynamic type CMulticlassSOLabels\n")

	return labels;
}
