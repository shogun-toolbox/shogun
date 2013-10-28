/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LABELS__H__
#define _LABELS__H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/labels/LabelsFactory.h>
#include <shogun/features/SubsetStack.h>

namespace shogun
{
/** @brief The class Labels models labels, i.e. class assignments of objects.
 *
 * Labels is the base class for general Label objects.
 *
 * Label objects need to overload get_num_labels() and is_valid()
 *
 * Labels support subset access is supported (partly).
 * Simple use the set_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method
 */
class CLabels : public CSGObject
{
	public:
		/** default constructor */
		CLabels();

		/** destructor */
		virtual ~CLabels();

		/** Make sure the label is valid, otherwise raise SG_ERROR.
		 *
		 * possible with subset
         *
         * @param context optional message to convey the context
		 */
		virtual void ensure_valid(const char* context=NULL)=0;

		/** get number of labels, depending on whether a subset is set
		 *
		 * @return number of labels
		 */
		virtual int32_t get_num_labels() const=0;

		/** get label type
		 *
		 * @return label type (binary, multiclass, ...)
		 */
		virtual ELabelType get_label_type() const=0;

		/** adds a subset of indices on top of the current subsets (possibly
		 * subset of subset. Calls subset_changed_post() afterwards
		 *
		 * @param subset subset of indices to add
		 * */
		virtual void add_subset(SGVector<index_t> subset);

		/** removes that last added subset from subset stack, if existing
		 * Calls subset_changed_post() afterwards */
		virtual void remove_subset();

		/** removes all subsets
		 * Calls subset_changed_post() afterwards */
		virtual void remove_all_subsets();

		/** set the confidence value for a particular label
		 *
		 * @param value value to set
		 * @param idx label index whose conf. value is to be changed
		 */
		virtual void set_value(float64_t value, int32_t idx);

		/** get confidence value for a particular label
		 *
		 * @param idx label index
		 * @return confidence value of label with index idx
		 */
		virtual float64_t get_value(int32_t idx);

		/** set confidence vector
		 *
		 * @param values to be set (should have zero length to disable
		 * values or length must match the number of labels)
		 */
		virtual void set_values(SGVector<float64_t> values);

		/** get confidence vector
		 *
		 * @return confidences
		 */
		virtual SGVector<float64_t> get_values();

	private:
		void init();

	protected:

		/** subset class to enable subset support for this class */
		CSubsetStack* m_subset_stack;

		/** current active value vector */
		SGVector<float64_t> m_current_values;
};
}
#endif
