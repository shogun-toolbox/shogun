#include <set>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/base/range.h>


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

CMulticlassLabels::CMulticlassLabels(const CMulticlassLabels& orig)
    : CDenseLabels(orig)
{
	init();
	m_multiclass_confidences = orig.m_multiclass_confidences;
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

bool CMulticlassLabels::is_valid() const
{
	// check labels are integers
	for (auto i : range(get_num_labels()))
	{
		auto label = get_label(i);

		if (label<0 || label != int64_t(label))
			return false;
	}

	// check labels are contiguous
	auto uniq = get_labels().unique();
	for (auto i : range(uniq.size()))
		if (i != uniq[i])
			return false;

	return true;
}

void CMulticlassLabels::ensure_valid(const char* context)
{
	REQUIRE(is_valid(), "Multiclass Labels must be in range "
	                    "[0,...,num_classes] and integers!\n");
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

int32_t CMulticlassLabels::get_num_classes()
{
	return get_labels().count_unique();
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

CLabels* CMulticlassLabels::duplicate() const
{
	return new CMulticlassLabels(*this);
}
