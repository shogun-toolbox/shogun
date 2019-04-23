/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Written (W) 2013 Thoralf Klein
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2013 Thoralf Klein and Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2014 Abinash Panda
 */

#ifndef _MULTILABEL_SO_LABELS__H__
#define _MULTILABEL_SO_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>
#include <shogun/labels/MultilabelLabels.h>

namespace shogun
{

class MultilabelSOLabels;

/** @brief Class CSparseMultilabel to be used in the application of Structured
 * Output (SO) learning to Multilabel classification.*/
class SparseMultilabel : public StructuredData
{
public:
	/** data type */
	STRUCTURED_DATA_TYPE(SDT_SPARSE_MULTILABEL);

	/** default constructor */
	SparseMultilabel() { }

	/** constructor
	 *
	 * @param label sparse label
	 */
	SparseMultilabel(SGVector<int32_t> label) : StructuredData(), m_label(label) { }

	/** destructor */
	~SparseMultilabel() { }

	/** helper method used to specialize a base class instance
	 *
	 * @param base_data its dynamic type must be CSparseMultilabel
	 */
	static std::shared_ptr<SparseMultilabel > obtain_from_generic(std::shared_ptr<StructuredData > base_data)
	{
		if (base_data->get_structured_data_type() == SDT_SPARSE_MULTILABEL)
		{
			return std::static_pointer_cast<SparseMultilabel>(base_data);
		}
		else
		{
			error("base_data must be of dynamic type CSparseMultilabel");
		}

		return NULL;
	}

	/** @return name of SGSerializable */
	virtual const char * get_name() const
	{
		return "SparseMultilabel";
	}

	/** @return label */
	SGVector<int32_t> get_data() const
	{
		return m_label;
	}

protected:
	/** sparse label */
	SGVector<int32_t> m_label;
}; /* class CSparseMultilabel */

/** @brief Class CMultilabelSOLabels used in the application of Structured
 * Output (SO) learning to Multilabel Classification. Labels are subsets
 * of {0, 1, ..., num_classes-1}. Each of the label if of type CSparseMultilabel and
 * all of them are stored in a DynamicObjectArray.
 */
class MultilabelSOLabels : public StructuredLabels
{
public:
	/** default constructor */
	MultilabelSOLabels();

	/** constructor
	 *
	 * @param num_classes number of (binary) class assignment per label
	 */
	MultilabelSOLabels(int32_t num_classes);

	/** constructor
	 *
	 * @param num_labels number of labels
	 * @param num_classes number of (binary) class assignment per label
	 */
	MultilabelSOLabels(int32_t num_labels, int32_t num_classes);

	/** constructor
	 *
	 * @param multilabel_labels
	 */
	MultilabelSOLabels(std::shared_ptr<MultilabelLabels > multilabel_labels);

	/** destructor */
	~MultilabelSOLabels();

	/** @return name of the SGSerializable */
	virtual const char * get_name() const
	{
		return "MultilabelSOLabels";
	}

	/** @return number of stored labels */
	virtual int32_t get_num_labels() const;

	/** @return number of classes (per label) */
	virtual int32_t get_num_classes() const;

	/** @return multilabel-labels object */
	virtual std::shared_ptr<MultilabelLabels > get_multilabel_labels();

	/** set sparse labels
	 *
	 * @param labels list of sparse labels
	 */
	virtual void set_sparse_labels(SGVector<int32_t> * labels);

	/** set sparse assignment for j-th label
	 *
	 * @param j label index
	 * @param label sparse label
	 */
	virtual void set_sparse_label(int32_t j, SGVector<int32_t> label);

	/** set assignment for j-th label
	 *
	 * @param j label index
	 * @param label sparse label
	 */
	virtual bool set_label(int32_t j, std::shared_ptr<StructuredData > label);

	/** add a new label to the vector of labels.
	 * This method should be used when inserting labels for the first time.
	 *
	 * @param label sparse label to add
	 */
	virtual void add_label(std::shared_ptr<StructuredData > label);

	/** get sparse assigment for j-th label
	 *
	 * @param j label index
	 */
	virtual SGVector<int32_t> get_sparse_label(int32_t j);

	/** get label for j-th index
	 *
	 * @param j label index
	 */
	virtual std::shared_ptr<StructuredData > get_label(int32_t j);

	/** Make sure the label is valid, otherwise raise SG_ERROR
	 *
	 * @param context optional message to convey the context
	 */
	virtual void ensure_valid(const char * context = NULL);

	/** Convert sparse labels to dense. The dense vector would be {d_true;
	 * d_false}^dense_dim. Indices in sparse would be marked "d_true",
	 * everything else "d_false".
	 *
	 * @param label sparse label to convert
	 * @param dense_dim dense dimension
	 * @param d_true marker for "true" labels
	 * @param d_false marker for "false" labels
	 *
	 * @return SGVector<float64_t> dense vector of dimension dense_dim
	 */
	static SGVector<float64_t> to_dense(std::shared_ptr<StructuredData > label,
	                                    int32_t dense_dim, float64_t d_true, float64_t d_false);

private:
	std::shared_ptr<MultilabelLabels > m_multilabel_labels;
	int32_t m_last_set_label;

private:
	void init();

}; /* class CMultilabelSOLabels */

} /* namespace shogun */

#endif /* _MULTILABEL_SO_LABELS__H__ */
