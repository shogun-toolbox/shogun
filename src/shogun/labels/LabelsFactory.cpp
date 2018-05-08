#include <shogun/labels/LabelsFactory.h>

#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/LatentLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/labels/MultilabelLabels.h>
#include <shogun/structure/MulticlassSOLabels.h>

using namespace shogun;

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
