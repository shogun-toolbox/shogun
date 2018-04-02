#include <set>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/MulticlassLabels.h>

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

CMulticlassLabels::CMulticlassLabels(CBinaryLabels* labels)
    : CDenseLabels(labels->get_num_labels())
{
	init();

	for (index_t i = 0; i < labels->get_num_labels(); ++i)
		m_labels[i] = (labels->get_label(i) == 1 ? 1 : 0);
}

CMulticlassLabels::~CMulticlassLabels()
{
}

void CMulticlassLabels::init()
{
	m_multiclass_confidences=SGMatrix<float64_t>();
}

void CMulticlassLabels::set_multiclass_confidences(int32_t i,
		SGVector<float64_t> confidences)
{
	REQUIRE(confidences.size()==m_multiclass_confidences.num_rows,
			"%s::set_multiclass_confidences(): Length of confidences should "
			"match size of the matrix", get_name());

	m_multiclass_confidences.set_column(i, confidences);
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

SGVector<float64_t> CMulticlassLabels::get_confidences_for_class(int32_t i)
{
	REQUIRE(
	    (m_multiclass_confidences.num_rows != 0) &&
	        (m_multiclass_confidences.num_cols != 0),
	    "Empty confidences, which need to be allocated before fetching.\n");

	SGVector<float64_t> confs(m_multiclass_confidences.num_cols);
	for (index_t j = 0; j < confs.size(); j++)
		confs[j] = m_multiclass_confidences(i, j);

	return confs;
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
	sg_memcpy(result.vector, unique_labels.vector,
			sizeof(float64_t)*unique_labels.vlen);

	return result;
}


int32_t CMulticlassLabels::get_num_classes()
{
	SGVector<float64_t> unique=get_unique_labels();
	return unique.vlen;
}

CLabels* CMulticlassLabels::shallow_subset_copy()
{
	CLabels* shallow_copy_labels=NULL;
	SGVector<float64_t> shallow_copy_vector(m_labels);
	shallow_copy_labels=new CMulticlassLabels(m_labels.size());
	SG_REF(shallow_copy_labels);
	((CDenseLabels*) shallow_copy_labels)->set_labels(shallow_copy_vector);
	if (m_subset_stack->has_subsets())
		shallow_copy_labels->add_subset(m_subset_stack->get_last_subset()->get_subset_idx());

	return shallow_copy_labels;
}

CMulticlassLabels* CMulticlassLabels::obtain_from_generic(CLabels* labels)
{
	if (labels == NULL)
		return NULL;

	if (labels->get_label_type() != LT_MULTICLASS)
	{
		SG_SERROR("The Labels passed cannot be casted to CMulticlassLabels!")
		return NULL;
	}

	CMulticlassLabels* casted = dynamic_cast<CMulticlassLabels*>(labels);
	SG_REF(casted)
	return casted;
}

namespace shogun
{
	SG_FORCED_INLINE Some<CMulticlassLabels> to_multiclass(CDenseLabels* orig)
	{
		auto int_labels = orig->get_int_labels();
		std::set<int32_t> unique(int_labels.begin(), int_labels.end());
		// TODO: anything discrete should be a possible multiclasslabels.
		// Once that is the case, this check can be removed
		// For now, if we don't enforce this, there will be crashes as the
		// class is used to index vectors
		REQUIRE(
		    (*std::min_element(unique.begin(), unique.end())) == 0 &&
		        (*std::max_element(unique.begin(), unique.end())) ==
		            (index_t)unique.size() - 1,
		    "Multiclass labels must be contiguous integers in [0, ..., "
		    "num_classes -1].\n");

		return some<CMulticlassLabels>(orig->get_labels());
	}

	Some<CMulticlassLabels> multiclass_labels(CLabels* orig)
	{
		REQUIRE(orig, "No labels provided.\n");
		try
		{
			switch (orig->get_label_type())
			{
			case LT_MULTICLASS:
				return Some<CMulticlassLabels>::from_raw(
				    orig->as<CMulticlassLabels>());
			case LT_DENSE_GENERIC:
				return to_multiclass(orig->as<CDenseLabels>());
			default:
				SG_SNOTIMPLEMENTED
			}
		}
		catch (const ShogunException& e)
		{
			SG_SERROR(
			    "Cannot convert %s to multiclass labels: %s\n",
			    orig->get_name(), e.what());
		}

		return Some<CMulticlassLabels>::from_raw(nullptr);
	}
} // namespace shogun
