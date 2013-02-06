#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

CMulticlassLabels::CMulticlassLabels() : CDenseLabels()
{
	m_multiclass_confidences = NULL;
	m_num_multiclass_confidences = 0;
}

CMulticlassLabels::CMulticlassLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
	m_multiclass_confidences = SG_MALLOC(SGVector<float64_t>, num_labels);
	m_num_multiclass_confidences = num_labels;
}

CMulticlassLabels::CMulticlassLabels(const SGVector<float64_t> src) : CDenseLabels()
{
	set_labels(src);
	m_multiclass_confidences = NULL;
	m_num_multiclass_confidences = 0;
}

CMulticlassLabels::CMulticlassLabels(CFile* loader) : CDenseLabels(loader)
{
	m_multiclass_confidences = NULL;
	m_num_multiclass_confidences = 0;
}

CMulticlassLabels::~CMulticlassLabels()
{
	SG_FREE(m_multiclass_confidences);
}

void CMulticlassLabels::set_multiclass_confidences(int32_t i, SGVector<float64_t> confidences)
{
	m_multiclass_confidences[i] = confidences;
}

SGVector<float64_t> CMulticlassLabels::get_multiclass_confidences(int32_t i)
{
	return m_multiclass_confidences[i];
}

CMulticlassLabels* CMulticlassLabels::obtain_from_generic(CLabels* base_labels)
{
	if ( base_labels->get_label_type() == LT_MULTICLASS )
		return (CMulticlassLabels*) base_labels;
	else
		SG_SERROR("base_labels must be of dynamic type CMulticlassLabels")

	return NULL;
}

void CMulticlassLabels::ensure_valid(const char* context)
{       
    CDenseLabels::ensure_valid(context);

    int32_t subset_size=get_num_labels();
    for (int32_t i=0; i<subset_size; i++)
    {
        int32_t real_i = m_subset_stack->subset_idx_conversion(i);
        int32_t label = int32_t(m_labels[real_i]);

        if (label<0 || float64_t(label)!=m_labels[real_i])
		{
			SG_ERROR("%s%sMulticlass Labels must be in range 0...<nr_classes-1> and integers!\n",
                    context?context:"", context?": ":"");
		}
	}
}

ELabelType CMulticlassLabels::get_label_type()
{
	return LT_MULTICLASS;
}

CBinaryLabels* CMulticlassLabels::get_binary_for_class(int32_t i)
{
	SGVector<float64_t> binary_labels(get_num_labels());

	bool use_confidences = false;
	if (m_num_multiclass_confidences != 0)
	{
		if (m_multiclass_confidences[i].size())
			use_confidences = true;
	}
	if (use_confidences)
	{
		for (int32_t k=0; k<binary_labels.vlen; k++)
		{
			SGVector<float64_t> confs = m_multiclass_confidences[k];
			int32_t label = get_int_label(k);
			binary_labels[k] = label == i ? confs[label] : -confs[label];
		}
	}
	else
	{
		for (int32_t k=0; k<binary_labels.vlen; k++)
		{
			int32_t label = get_int_label(k);
			binary_labels[k] = label == i ? +1.0 : -1.0;
		}
	}
	return new CBinaryLabels(binary_labels);
}

SGVector<float64_t> CMulticlassLabels::get_unique_labels()
{
	/* extract all labels (copy because of possible subset) */
	SGVector<float64_t> unique_labels=get_labels_copy();
	unique_labels.vlen=SGVector<float64_t>::unique(unique_labels.vector, unique_labels.vlen);

	SGVector<float64_t> result(unique_labels.vlen);
	memcpy(result.vector, unique_labels.vector,
			sizeof(float64_t)*unique_labels.vlen);

	return result;
}


int32_t CMulticlassLabels::get_num_classes()
{
	SGVector<float64_t> unique=get_unique_labels();
	return unique.vlen;
}
