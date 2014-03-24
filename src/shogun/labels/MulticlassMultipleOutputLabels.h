/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef MULTICLASSMULTIPLEOUTPUTLABELS_H_
#define MULTICLASSMULTIPLEOUTPUTLABELS_H_

#include <shogun/lib/config.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
/** @brief Multiclass Labels for multi-class classification
 * with multiple labels
 *
 * valid values for labels are 0...nr_classes-1
 *
 * Each label in this setting is vector of a few labels
 */
class CMulticlassMultipleOutputLabels : public CLabels
{

	public:
		/** default constructor */
		CMulticlassMultipleOutputLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CMulticlassMultipleOutputLabels(int32_t num_labels);

		/** destructor */
		virtual ~CMulticlassMultipleOutputLabels();

		/** check if labeling is valid
		 *
		 * possible with subset
		 *
		 * @return if labeling is valid
		 */
		virtual void ensure_valid(const char* context = NULL);

		/** get labels
		 *
		 * not possible with subset
		 *
		 * @return labels
		 */
		SGMatrix<index_t> get_labels() const;

		/** get label object for specified index
		 *
		 * @param idx index of the label
		 *
		 * @return label object
		 */
		SGVector<index_t> get_label(int32_t idx);

		/**
		 * set label, possible with subset. This method should be used
		 * when substituting labels previously inserted. To insert new
		 * labels, use the method add_label.
		 *
		 * @param idx index of label to set
		 * @param label value of label
		 *
		 * @return if setting was successful
		 */
		bool set_label(int32_t idx, SGVector<index_t> label);

		/** get number of labels, depending on wheter a subset is set
		 *
		 * @return number of labels
		 */
		virtual int32_t get_num_labels() const;

		/** @return object name */
		virtual const char* get_name() const { return "MulticlassMultipleOutputLabels"; }

		/** get label type
		 *
		 * @return label type LT_STRUCTURED
		 */
		virtual ELabelType get_label_type() const { return LT_MULTICLASS_MULTIPLE_OUTPUT; }

	private:
		/** internal initialization */
		void init();

	protected:
		/** vector of labels */
		SGVector<index_t>* m_labels;
		/** number of labels */
		int32_t m_n_labels;

};
}
#endif
