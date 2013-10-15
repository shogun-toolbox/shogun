#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/base/ParameterMap.h>

using namespace shogun;

CMulticlassLabels::CMulticlassLabels() : CDenseLabels()
{
	init();
}

CMulticlassLabels::CMulticlassLabels(int32_t num_labels) : CDenseLabels(num_labels)
{
	init();
}

CMulticlassLabels::CMulticlassLabels(const SGVector<float64_t> src) : CDenseLabels()
{
	init();
	set_labels(src);
}

CMulticlassLabels::CMulticlassLabels(CFile* loader) : CDenseLabels(loader)
{
	init();
}

CMulticlassLabels::~CMulticlassLabels()
{
}

void CMulticlassLabels::init()
{
	/* for this to work, migration has to be fixed */
//	SG_ADD(&m_multiclass_confidences, "multiclass_confidences", "Vectors of "
//			"multiclass confidences", MS_NOT_AVAILABLE);

//	m_parameter_map->finalize_map();

	m_multiclass_confidences=SGMatrix<float64_t>();
}

void CMulticlassLabels::set_multiclass_confidences(int32_t i,
		SGVector<float64_t> confidences)
{
	REQUIRE(confidences.size()==m_multiclass_confidences.num_rows,
			"%s::set_multiclass_confidences(): Length of confidences should "
			"match size of the matrix", get_name());

	for (index_t j=0; j<confidences.size(); j++)
		m_multiclass_confidences(j,i) = confidences[j];
}

SGVector<float64_t> CMulticlassLabels::get_multiclass_confidences(int32_t i)
{
	SGVector<float64_t> confs(m_multiclass_confidences.num_rows);
	for (index_t j=0; j<confs.size(); j++)
		confs[j] = m_multiclass_confidences(j,i);

	return confs;
}

void CMulticlassLabels::allocate_confidences_for(int32_t n_classes)
{
	int32_t n_labels = m_labels.size();
	REQUIRE(n_labels!=0,"%s::allocate_confidences_for(): There should be "
			"labels to store confidences", get_name());

	m_multiclass_confidences = SGMatrix<float64_t>(n_classes,n_labels);
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

ELabelType CMulticlassLabels::get_label_type() const
{
	return LT_MULTICLASS;
}

CBinaryLabels* CMulticlassLabels::get_binary_for_class(int32_t i)
{
	SGVector<float64_t> binary_labels(get_num_labels());

	bool use_confidences = false;
	if ((m_multiclass_confidences.num_rows != 0) && (m_multiclass_confidences.num_cols != 0))
	{
		use_confidences = true;
	}
	if (use_confidences)
	{
		for (int32_t k=0; k<binary_labels.vlen; k++)
		{
			int32_t label = get_int_label(k);
			float64_t confidence = m_multiclass_confidences(label,k);
			binary_labels[k] = label == i ? confidence : -confidence;
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
