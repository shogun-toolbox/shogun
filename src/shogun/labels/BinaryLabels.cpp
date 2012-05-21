#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

CBinaryLabels::CBinaryLabels() : CDenseLabels(), m_threshold(0.0)
{
}

CBinaryLabels::CBinaryLabels(int32_t num_labels) : CDenseLabels(num_labels),
	m_threshold(0.0)
{
}

CBinaryLabels::CBinaryLabels(SGVector<float64_t> src) : CDenseLabels(),
	m_threshold(0.0)
{
	SGVector<float64_t> labels(src.vlen);
	for (int32_t i=0; i<labels.vlen; i++)
		labels[i] = CMath::sign(src[i]+m_threshold);
	set_labels(labels);
	set_confidences(src);
}

CBinaryLabels::CBinaryLabels(CFile* loader) : CDenseLabels(loader),
	m_threshold(0.0)
{
}

CBinaryLabels* CBinaryLabels::obtain_from_generic(CLabels* base_labels)
{
	if ( base_labels->get_label_type() == LT_BINARY )
		return (CBinaryLabels*) base_labels;
	else
		SG_ERROR("base_labels must be of dynamic type CBinaryLabels");

	return NULL;
}


bool CBinaryLabels::is_valid()
{       
    ASSERT(m_labels.vector);
    bool found_plus_one=false;
    bool found_minus_one=false;

    int32_t subset_size=get_num_labels();
    for (int32_t i=0; i<subset_size; i++)
    {
        int32_t real_i=m_subset_stack->subset_idx_conversion(i);
        if (m_labels[real_i]==+1.0)
            found_plus_one=true;
        else if (m_labels[real_i]==-1.0)
            found_minus_one=true;
        else
        {
            SG_ERROR("Not a two class labeling label[%d]=%f (only +1/-1 "
                    "allowed)\n", i, m_labels[real_i]);
        }
    }
    
    if (!found_plus_one)
        SG_ERROR("Not a two class labeling - no positively labeled examples found\n");
    if (!found_minus_one)
        SG_ERROR("Not a two class labeling - no negatively labeled examples found\n");

    return true;
}

ELabelType CBinaryLabels::get_label_type()
{
	return LT_BINARY;
}
