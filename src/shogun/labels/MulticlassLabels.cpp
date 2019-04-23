#include <set>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

MulticlassLabels::MulticlassLabels() : DenseLabels()
{
	init();
}

MulticlassLabels::MulticlassLabels(int32_t num_labels) : DenseLabels(num_labels)
{
	init();
}

MulticlassLabels::MulticlassLabels(const SGVector<float64_t> src) : DenseLabels()
{
	init();
	set_labels(src);
}

MulticlassLabels::MulticlassLabels(std::shared_ptr<File> loader) : DenseLabels(loader)
{
	init();
}

MulticlassLabels::MulticlassLabels(std::shared_ptr<BinaryLabels> labels)
    : DenseLabels(labels->get_num_labels())
{
	init();

	for (index_t i = 0; i < labels->get_num_labels(); ++i)
		m_labels[i] = (labels->get_label(i) == 1 ? 1 : 0);
}

MulticlassLabels::MulticlassLabels(const MulticlassLabels& orig)
    : DenseLabels(orig)
{
	init();
	m_multiclass_confidences = orig.m_multiclass_confidences;
}

MulticlassLabels::~MulticlassLabels()
{
}

void MulticlassLabels::init()
{
	m_multiclass_confidences=SGMatrix<float64_t>();
}

void MulticlassLabels::set_multiclass_confidences(int32_t i,
		SGVector<float64_t> confidences)
{
	require(confidences.size()==m_multiclass_confidences.num_rows,
			"{}::set_multiclass_confidences(): Length of confidences should "
			"match size of the matrix", get_name());

	m_multiclass_confidences.set_column(i, confidences);
}

SGVector<float64_t> MulticlassLabels::get_multiclass_confidences(int32_t i)
{
	SGVector<float64_t> confs(m_multiclass_confidences.num_rows);
	for (index_t j=0; j<confs.size(); j++)
		confs[j] = m_multiclass_confidences(j,i);

	return confs;
}

void MulticlassLabels::allocate_confidences_for(int32_t n_classes)
{
	int32_t n_labels = m_labels.size();
	require(n_labels!=0,"{}::allocate_confidences_for(): There should be "
			"labels to store confidences", get_name());

	m_multiclass_confidences = SGMatrix<float64_t>(n_classes,n_labels);
}

SGVector<float64_t> MulticlassLabels::get_confidences_for_class(int32_t i)
{
	require(
	    (m_multiclass_confidences.num_rows != 0) &&
	        (m_multiclass_confidences.num_cols != 0),
	    "Empty confidences, which need to be allocated before fetching.");

	SGVector<float64_t> confs(m_multiclass_confidences.num_cols);
	for (index_t j = 0; j < confs.size(); j++)
		confs[j] = m_multiclass_confidences(i, j);

	return confs;
}

bool MulticlassLabels::is_valid() const
{
	if (!DenseLabels::is_valid())
		return false;

	int32_t subset_size=get_num_labels();
    for (int32_t i=0; i<subset_size; i++)
    {
        int32_t real_i = m_subset_stack->subset_idx_conversion(i);
		int32_t label = int64_t(m_labels[real_i]);

		if (label<0 || float64_t(label)!=m_labels[real_i])
		{
			return false;
		}
	}
	return true;
}

void MulticlassLabels::ensure_valid(const char* context)
{
	require(is_valid(), "Multiclass Labels must be in range "
	                    "[0,...,num_classes] and integers!");
}

ELabelType MulticlassLabels::get_label_type() const
{
	return LT_MULTICLASS;
}

std::shared_ptr<BinaryLabels> MulticlassLabels::get_binary_for_class(int32_t i)
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
	return std::make_shared<BinaryLabels>(binary_labels);
}

SGVector<float64_t> MulticlassLabels::get_unique_labels()
{
	/* extract all labels (copy because of possible subset) */
	SGVector<float64_t> unique_labels=get_labels_copy();
	unique_labels.vlen=SGVector<float64_t>::unique(unique_labels.vector, unique_labels.vlen);

	SGVector<float64_t> result(unique_labels.vlen);
	sg_memcpy(result.vector, unique_labels.vector,
			sizeof(float64_t)*unique_labels.vlen);

	return result;
}


int32_t MulticlassLabels::get_num_classes()
{
	SGVector<float64_t> unique=get_unique_labels();
	return unique.vlen;
}

std::shared_ptr<Labels> MulticlassLabels::shallow_subset_copy()
{
	SGVector<float64_t> shallow_copy_vector(m_labels);
	auto shallow_copy_labels=std::make_shared<MulticlassLabels>(m_labels.size());

	shallow_copy_labels->set_labels(shallow_copy_vector);
	if (m_subset_stack->has_subsets())
		shallow_copy_labels->add_subset(m_subset_stack->get_last_subset()->get_subset_idx());

	return shallow_copy_labels;
}

std::shared_ptr<MulticlassLabels> MulticlassLabels::obtain_from_generic(std::shared_ptr<Labels> labels)
{
	if (labels == NULL)
		return NULL;

	if (labels->get_label_type() != LT_MULTICLASS)
	{
		error("The Labels passed cannot be casted to MulticlassLabels!");
		return NULL;
	}

	return std::dynamic_pointer_cast<MulticlassLabels>(labels);
}

std::shared_ptr<Labels> MulticlassLabels::duplicate() const
{
	return std::make_shared<MulticlassLabels>(*this);
}

namespace shogun
{
	SG_FORCED_INLINE std::shared_ptr<MulticlassLabels> to_multiclass(std::shared_ptr<DenseLabels> orig)
	{
		auto result_vector = orig->get_labels();
		std::set<int32_t> unique(result_vector.begin(), result_vector.end());
		// potentially convert to [0,1, ..., num_classes-1] if not in that form
		// TODO: remove this once multiclass labels can be any discrete set
		auto min = (*std::min_element(unique.begin(), unique.end()));
		auto max = (*std::max_element(unique.begin(), unique.end()));
		if (!(min == 0 && max == (index_t)unique.size() - 1))
		{
			// print conversion table for users
			io::warn(
			    "Converting non-contiguous multiclass labels to "
			    "contiguous version:",
			    unique.size() - 1);
			std::for_each(
			    unique.begin(), unique.end(), [&unique](int32_t old_label) {
				    auto new_label =
				        std::distance(unique.begin(), unique.find(old_label));
				    io::warn("Converting {} to {}.", old_label, new_label);
				});

			SGVector<float64_t> converted(result_vector.vlen);
			std::transform(
			    result_vector.begin(), result_vector.end(), converted.begin(),
			    [&unique](int32_t old_label) {
				    return std::distance(
				        unique.begin(), unique.find(old_label));
				});
			result_vector = converted;
		}
		return std::make_shared<MulticlassLabels>(result_vector);
	}

	std::shared_ptr<MulticlassLabels> multiclass_labels(std::shared_ptr<Labels> orig)
	{
		require(orig, "No labels provided.");
		try
		{
			switch (orig->get_label_type())
			{
			case LT_MULTICLASS:
				return std::static_pointer_cast<MulticlassLabels>(orig);
			case LT_BINARY:
				return to_multiclass(std::static_pointer_cast<BinaryLabels>(orig));
			default:
				not_implemented(SOURCE_LOCATION);
			}
		}
		catch (const ShogunException& e)
		{
			error(
			    "Cannot convert {} to multiclass labels: {}",
			    orig->get_name(), e.what());
		}

		return nullptr;
	}
} // namespace shogun
