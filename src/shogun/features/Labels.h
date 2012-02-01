/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Subset support written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LABELS__H__
#define _LABELS__H__

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/Subset.h>

namespace shogun
{
	class CFile;

/** @brief The class Labels models labels, i.e. class assignments of objects.
 *
 * Labels here are always real-valued and thus applicable to classification
 * (cf.  CClassifier) and regression (cf. CRegression) problems.
 *
 * (Partly) subset access is supported.
 * Simple use the set_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method
 */
class CLabels : public CSGObject
{
	public:
		/** default constructor */
		CLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 */
		CLabels(SGVector<float64_t> src);

		/* constructor
		 * 
		 * @param labels labels 
		 */
		//CLabels(SGVector<int64_t> labels);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CLabels(CFile* loader);

		/** destructor */
		virtual ~CLabels();

		/** load labels from file
		 *
		 * any subset is removed before
		 *
		 * @param loader File object via which to load data
		 */
		virtual void load(CFile* loader);

		/** save labels to file
		 *
		 * not possible with subset
		 *
		 * @param writer File object via which to save data
		 */
		virtual void save(CFile* writer);

		/** set label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to set
		 * @param label value of label
		 * @return if setting was successful
		 */
		bool set_label(int32_t idx, float64_t label);

		/** set INT label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to set
		 * @param label INT value of label
		 * @return if setting was successful
		 */
		bool set_int_label(int32_t idx, int32_t label);

		/** get label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to get
		 * @return value of label
		 */
		float64_t get_label(int32_t idx);

		/** get INT label
		 *
		 * possible with subset
		 *
		 * @param idx index of label to get
		 * @return INT value of label
		 */
		int32_t get_int_label(int32_t idx);

		/** is two-class labeling
		 *
		 * possible with subset
		 *
		 * @return if this is two-class labeling
		 */
		bool is_two_class_labeling();

		/** return number of classes (for multiclass)
		 *
		 * possible with subset
		 *
		 * @return number of classes
		 */
		int32_t get_num_classes();

		/** get labels
		 *
		 * not possible with subset
		 *
		 * @return labels
		 */
		SGVector<float64_t> get_labels();

		/** get copy of labels. Caller has to clean up
		 *
		 * possible with subset
		 *
		 * @return labels
		 */
		SGVector<float64_t> get_labels_copy();

		/** get unqiue labels (new SGVector, caller has to clean up)
		 *
		 * possible with subset
		 *
		 * @return unique labels
		 */
		SGVector<float64_t> get_unique_labels();

		/** set labels
		 *
		 * not possible with subset
		 *
		 * @param v labels
		 */
		void set_labels(SGVector<float64_t> v);

		/**
		 * set all labels to +1
		 *
		 * possible with subset
		 * */
		void set_to_one();

		/** get INT label vector
		 * caller has to clean up
		 *
		 * possible with subset
		 *
		 * @return INT labels
		 */
		SGVector<int32_t> get_int_labels();

		/** set INT labels
		 * caller has to clean up
		 *
		 * not possible on subset
		 *
		 * @param labels INT labels
		 */
		void set_int_labels(SGVector<int32_t> labels);

		/** get number of labels, depending on whether a subset is set
		 *
		 * @return number of labels
		 */
		int32_t get_num_labels();

		/** @return object name */
		inline virtual const char* get_name() const { return "Labels"; }

		/** setter for subset variable, deletes old one
		 *
		 * @param subset subset instance to set
		 */
		virtual void set_subset(CSubset* subset);

		/** deletes any set subset */
		virtual void remove_subset();

		/** does subset index conversion with the underlying subset if possible
		 *
		 * @param idx index to convert
		 * @return converted index
		 */
		index_t subset_idx_conversion(index_t idx) const;

	private:
		void init();

	protected:
		/** the label vector */
		SGVector<float64_t> labels;

		/** number of classes */
		int32_t m_num_classes;

	private:
		/* subset class to enable subset support for this class */
		CSubset* m_subset;
};
}
#endif
