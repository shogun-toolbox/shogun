#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

CBinaryLabels::CBinaryLabels() : CDenseLabels()
{
}

CBinaryLabels::CBinaryLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
}

CBinaryLabels::CBinaryLabels(SGVector<float64_t> src, float64_t threshold) : CDenseLabels()
{
	SGVector<float64_t> labels(src.vlen);
	for (int32_t i=0; i<labels.vlen; i++)
		labels[i] = src[i]+threshold>=0 ? +1.0 : -1.0;
	set_labels(labels);
	set_confidences(src);
}

CBinaryLabels::CBinaryLabels(CFile* loader) : CDenseLabels(loader)
{
}

CBinaryLabels* CBinaryLabels::obtain_from_generic(CLabels* base_labels)
{
	if ( base_labels->get_label_type() == LT_BINARY )
		return (CBinaryLabels*) base_labels;
	else
		SG_SERROR("base_labels must be of dynamic type CBinaryLabels");

	return NULL;
}


void CBinaryLabels::ensure_valid(const char* context)
{
    CDenseLabels::ensure_valid(context);
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
            SG_ERROR("%s%sNot a two class labeling label[%d]=%f (only +1/-1 "
                    "allowed)\n", context?context:"", context?": ":"", i, m_labels[real_i]);
        }
    }
    
    if (!found_plus_one)
    {
        SG_ERROR("%s%sNot a two class labeling - no positively labeled examples found\n",
                context?context:"", context?": ":"");
    }

    if (!found_minus_one)
    {
        SG_ERROR("%s%sNot a two class labeling - no negatively labeled examples found\n",
                context?context:"", context?": ":"");
    }
}

ELabelType CBinaryLabels::get_label_type()
{
	return LT_BINARY;
}
