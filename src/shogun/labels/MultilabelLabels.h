/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013-2014 Thoralf Klein
 * Copyright (C) 2013 Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2013-2014 Thoralf Klein
 */

#ifndef _MULTILABEL_LABELS__H__
#define _MULTILABEL_LABELS__H__

#include <shogun/lib/common.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief Multilabel Labels for multi-label classification
 *
 * valid values for labels are subsets of {0, ..., num_classes-1}
 */

typedef int16_t mclass_t;

class CMultilabelLabels : public CLabels
{
	public:
		/** default constructor */
		CMultilabelLabels();

		/** constructor
		 *
		 * @param num_classes number of (binary) class assignments per label
		 */
		CMultilabelLabels(mclass_t num_classes);

		/** constructor
		 *
		 * @param num_labels  number of labels
		 * @param num_classes number of (binary) class assignments per label
		 */
		CMultilabelLabels(int32_t num_labels, mclass_t num_classes);

		/** destructor */
		~CMultilabelLabels();

		/** Make sure the label is valid, otherwise raise SG_ERROR.
		 *
		 * possible with subset
		 *
		 * @param context optional message to convey the context
		 */
		void ensure_valid(const char* context=NULL);

		/** get label type
		 *
		 * @return label type multiclass
		 */
		virtual ELabelType get_label_type() const { return LT_SPARSE_MULTILABEL; }

		/** @return object name */
		virtual const char* get_name() const { return "MultilabelLabels"; }

		/** get the number of stored labels
		 *
		 * @return the number of labels
		 */
		virtual int32_t get_num_labels() const;

		/** return number of classes (per label)
		 *
		 * @return number of classes
		 */
		virtual mclass_t get_num_classes() const;

		/** set labels
		 *
		 * @param labels list of sparse labels
		 */
		void set_labels(SGVector<mclass_t> * labels);

		/** get list of sparse class labels (one vector per class)
		 *
		 * @return SGVector<int32_t> **
		 */
		SGVector<int32_t> ** get_class_labels() const;

		/** get sparse assignment for j-th label
		 *
		 * @return SGVector<mclass_t> sparse label
		 */
		SGVector<mclass_t> get_label(int32_t j);

		/** Convert sparse label vector to dense.  The dense vector
		  * will be {d_true; d_false}^dense_dim.  Indices in sparse
		  * will be marked "d_true", everything else "d_false".
		  *
		  * @param SGVector<S> * sparse vector to convert
		  * @param int32_t       dense dimension
		  * @param D             marker for "true" labels
		  * @param D             marker for "false" labels
		  *
		  * @return SGVector<D> dense vector of dimension dense_len
		  */
		template <class S, class D>
		static SGVector<D> to_dense(SGVector<S> * sparse, int32_t dense_len, D d_true, D d_false);

		/** set sparse assignment for j-th label
		 *
		 * @param int32_t label index
		 * @param SGVector<mclass_t> sparse label
		 */
		void set_label(int32_t j, SGVector<mclass_t> label);

		/** TODO */
		void set_class_labels(SGVector <int32_t> ** labels_list);

		/** print the current labeling */
		void display() const;

		/** save labels to file */
		void save(const char *fname);

		/** read num_labels and num_classes from file fname */
		static void load_info(const char *fname, int32_t & num_labels, mclass_t & num_classes);

		/** @return CMultilabels * read from fname */
		static CMultilabelLabels * load(const char *fname);

	private:
		void init();

	protected:

		SGVector<mclass_t> * m_labels;
		int32_t              m_num_labels;

	public:
		mclass_t             m_num_classes;
};

}
#endif
