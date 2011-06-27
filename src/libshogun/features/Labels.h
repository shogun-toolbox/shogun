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

#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "base/SGObject.h"
#include "features/Subset.h"

namespace shogun
{

	class CFile;

/** @brief The class Labels models labels, i.e. class assignments of objects.
 *
 * Labels here are always real-valued and thus applicable to classification
 * (cf.  CClassifier) and regression (cf. CRegression) problems.
 *
 * Setting subsets is supported
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

		/** constructor
		 *
		 * @param src labels to set
		 * @param len number of labels
		 */
		CLabels(float64_t* src, int32_t len);

		/** constructor 
		 * @param in_confidences confidence matrix to be used to derive the labels
		 * @param in_num_labels number of labels
		 * @param in_num_classes number of classes
		 */
		CLabels(float64_t* in_confidences, int32_t in_num_labels, int32_t in_num_classes);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CLabels(CFile* loader);
		virtual ~CLabels();

		/** load labels from file
		 *
		 * @param loader File object via which to load data
		 */
		virtual void load(CFile* loader);

		/** save labels to file
		 *
		 * @param writer File object via which to save data
		 */
		virtual void save(CFile* writer);

		/** set label
		 *
		 * @param idx index of label to set
		 * @param label value of label
		 * @return if setting was successful
		 */
		inline bool set_label(int32_t idx, float64_t label)
		{
			int32_t real_num=m_subset->subset_idx_conversion(idx);
			if (labels && real_num<num_labels)
			{
				labels[real_num]=label;
				return true;
			}
			else 
				return false;
		}

		/** set INT label
		 *
		 * @param idx index of label to set
		 * @param label INT value of label
		 * @return if setting was successful
		 */
		inline bool set_int_label(int32_t idx, int32_t label)
		{ 
			int32_t real_num=m_subset->subset_idx_conversion(idx);
			if (labels && real_num<num_labels)
			{
				labels[real_num]= (float64_t) label;
				return true;
			}
			else 
				return false;
		}

		/** get label
		 *
		 * @param idx index of label to get
		 * @return value of label
		 */
		inline float64_t get_label(int32_t idx)
		{
			int32_t real_num=m_subset->subset_idx_conversion(idx);
			ASSERT(labels && real_num<num_labels);
			return labels[real_num];
		}

		/** get INT label
		 *
		 * @param idx index of label to get
		 * @return INT value of label
		 */
		inline int32_t get_int_label(int32_t idx)
		{
			int32_t real_num=m_subset->subset_idx_conversion(idx);
			ASSERT(labels && real_num<num_labels);
			ASSERT(labels[real_num]== ((float64_t) ((int32_t) labels[real_num])));
			return ((int32_t) labels[real_num]);
		}

		/** is two-class labeling
		 *
		 * @return if this is two-class labeling
		 */
		bool is_two_class_labeling();

		/** return number of classes (for multiclass)
		 * labels have to be zero based 0,1,...C missing
		 * labels are illegal
		 *
		 * @return number of classes
		 */
		int32_t get_num_classes();

		/** get labels
		 *
		 * @param len number of labels
		 * @return the labels
		 */
		float64_t* get_labels(int32_t &len);
		
		/** get labels
		 *
		 * @return labels
		 */
		SGVector<float64_t> get_labels();

		/** set labels
		 *
		 * @param v labels
		 */
		void set_labels(SGVector<float64_t> v);

		/** set labels
		 *
		 * @param src labels to set
		 * @param len number of labels
		 */
		void set_labels(float64_t* src, int32_t len);

		/** set all labels to +1 */
		void set_to_one();

		/** set confidences 
		 * @param in_confidences confidence matrix to be used to derive the labels
		 * @param in_num_labels number of labels
		 * @param in_num_classes number of classes
		 */
		void set_confidences(float64_t* in_confidences, int32_t in_num_labels, int32_t in_num_classes);

		/** get confidences 
		 * @param out_num_labels number of labels
		 * @param out_num_classes number of classes will be written to it
		 * @return pointer to the confidences matrix
		 */
		float64_t* get_confidences(int32_t& out_num_labels, int32_t& out_num_classes);

		/** get confidences  (swig compatible)
		 * @param dst pointer to the confidences matrix (returned)
		 * @param out_num_labels number of labels (returned)
		 * @param out_num_classes number of classes will be written to it (returned)
		 */
		void get_confidences(float64_t** dst, int32_t* out_num_labels, int32_t* out_num_classes);

		/** get confidences for a sample
		 * @param in_sample_index index of a sample
		 * @param out_num_classes number of classes will be written to it
		 * @return pointer to the confidences vector
		 */
		float64_t* get_sample_confidences(const int32_t& in_sample_index, int32_t& out_num_classes);

		/** get INT label vector
		 * caller has to clean up
		 *
		 * @param len number of labels to get
		 * @return INT labels
		 */
		int32_t* get_int_labels(int32_t &len);

		/** set INT labels
		 * caller has to clean up
		 *
		 * @param labels INT labels
		 * @param len number of INT labels
		 */
		void set_int_labels(int32_t *labels, int32_t len) ;

		/** get number of labels
		 *
		 * @return number of labels
		 */
		inline int32_t get_num_labels() { return num_labels; }

		/** @return object name */
		inline virtual const char* get_name() const { return "Labels"; }


		/*********************************
		 * wrapper for Subset methods
		 * (to avoid mutliple inheritance)
		 ********************************/
		virtual void remove_subset();

		virtual SGVector<index_t> get_subset() const
		{
			return m_subset->get_subset();
		}

		bool has_subset() const
		{
			return m_subset->has_subset();
		}

		virtual void set_subset(SGVector<index_t> subset);

		inline index_t subset_idx_conversion(index_t idx) const
		{
			return m_subset->subset_idx_conversion(idx);
		}
		/***********************************
		 * End of wrapper for Subset methods
		 **********************************/

	protected:
		/** find labels from the confidences using argmax over the classes. */
		void find_labels();

	private:
		void init();

	protected:
		/** number of labels */
		int32_t num_labels;
		/** the labels */
		float64_t* labels;

		/** number of classes */
		int32_t m_num_classes;


		/** confidence matrix of size: num_classes x num_labels */
		float64_t* m_confidences;
		int32_t m_confidence_classes;
		int32_t m_confidence_labels;

	private:
		/* subset class to enable subset support for this class */
		CSubset* m_subset;
};
}
#endif
