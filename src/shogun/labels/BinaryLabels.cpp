#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

CBinaryLabels::CBinaryLabels() : CDenseLabels()
{
}

CBinaryLabels::CBinaryLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
}

CBinaryLabels::CBinaryLabels(const SGVector<float64_t> src) : CDenseLabels(src)
{
}

CBinaryLabels::CBinaryLabels(CFile* loader) : CDenseLabels(loader)
{
}

bool CBinaryLabels::is_valid()
{       
    ASSERT(labels.vector);
    bool found_plus_one=false;
    bool found_minus_one=false;

    int32_t subset_size=get_num_labels();
    for (int32_t i=0; i<subset_size; i++)
    {
        int32_t real_i=m_subset_stack->subset_idx_conversion(i);
        if (labels.vector[real_i]==+1.0)
            found_plus_one=true;
        else if (labels.vector[real_i]==-1.0)
            found_minus_one=true;
        else
        {
            SG_ERROR("Not a two class labeling label[%d]=%f (only +1/-1 "
                    "allowed)\n", i, labels.vector[real_i]);
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
