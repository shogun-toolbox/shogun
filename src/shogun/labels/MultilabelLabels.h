/*
 * Copyright (C) 2013 Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2013-2014 Thoralf Klein
 * Written (W) 2013-2014 Thoralf Klein
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef _MULTILABEL_LABELS__H__
#define _MULTILABEL_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** @brief Multilabel Labels for multi-label classification
 *
 * Labels are subsets of {0, ..., num_classes-1}
 */

class MultilabelLabels : public Labels
{
public:
	/** default constructor */
	MultilabelLabels();

	/** constructor
	 *
	 * @param num_classes number of (binary) class assignments per label
	 */
	MultilabelLabels(int32_t num_classes);

	/** constructor
	 *
	 * @param num_labels  number of labels
	 * @param num_classes number of (binary) class assignments per label
	 */
	MultilabelLabels(int32_t num_labels, int32_t num_classes);

	/** destructor */
	~MultilabelLabels() override;

	bool is_valid() const override;

	/** Make sure the label is valid, otherwise raise SG_ERROR.
	 *
	 * possible with subset
	 *
	 * @param context optional message to convey the context
	 */
	void ensure_valid(const char * context = NULL) override;

	/** get label type
	 *
	 * @return label type multiclass
	 */
	ELabelType get_label_type() const override
	{
		return LT_SPARSE_MULTILABEL;
	}

	/** @return object name */
	const char * get_name() const override
	{
		return "MultilabelLabels";
	}

	/** get the number of stored labels
	 *
	 * @return the number of labels
	 */
	int32_t get_num_labels() const override;

	/** return number of classes (per label)
	 *
	 * @return number of classes
	 */
	virtual int32_t get_num_classes() const;

	/** set labels
	 *
	 * @param labels list of sparse labels
	 */
	void set_labels(SGVector<int32_t> * labels);

	/** get list of sparse class labels (one vector per class)
	 *
	 * @return list of sparse class labels
	 */
	SGVector<int32_t> ** get_class_labels() const;

	/** get label matrix, only for multiclass multiple output labels
	 *
	 * @return label matrix
	 */
	SGMatrix<int32_t> get_labels() const;

	/** get sparse assignment for j-th label
	 *
	 * @return sparse label
	 */
	SGVector<int32_t> get_label(int32_t j);

	/** Convert sparse label vector to dense.  The dense vector
	 * will be {d_true; d_false}^dense_dim.  Indices in sparse
	 * will be marked "d_true", everything else "d_false".
	 *
	 * @param sparse        sparse vector to convert
	 * @param dense_len     dimension
	 * @param d_true        marker for "true" labels
	 * @param d_false       marker for "false" labels
	 *
	 * @return dense vector of dimension dense_len
	 */
	template <class S, class D>
	static SGVector<D> to_dense(SGVector<S> * sparse, int32_t dense_len, D d_true, D d_false);

	/** set sparse assignment for j-th label
	 *
	 * @param j     label index
	 * @param label sparse label
	 */
	void set_label(int32_t j, SGVector<int32_t> label);

	/** assigning class labels */
	void set_class_labels(SGVector <int32_t> ** labels_list);

	/** print the current labeling */
	void display() const;

private:
	void init(int32_t num_labels, int32_t num_classes);

protected:
	/** Number of labels */
	int32_t m_num_labels;

	/** Number of classes */
	int32_t m_num_classes;

	/** Labels themselves */
	SGVector<int32_t>* m_labels;
};

}

#endif
