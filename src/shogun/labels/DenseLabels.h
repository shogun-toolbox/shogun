/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DENSE_LABELS__H__
#define _DENSE_LABELS__H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/labels/Labels.h>
#include <shogun/features/SubsetStack.h>

namespace shogun
{
	class CFile;

/** @brief Dense integer or floating point labels
 *
 * DenseLabels here are always real-valued and thus applicable to classification
 * (cf.  CClassifier) and regression (cf. CRegression) problems.
 *
 * This class implements the shared functions for storing, and accessing label
 * (vectors).
 */
class CDenseLabels : public CLabels
{
	public:
		/** default constructor */
		CDenseLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CDenseLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CDenseLabels(CFile* loader);

		/** destructor */
		virtual ~CDenseLabels();

		/** Make sure the label is valid, otherwise raise SG_ERROR.
		 *
		 * possible with subset
         *
         * @param context optional message to convey the context
		 */
		virtual void ensure_valid(const char* context=NULL);

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

		/** Getter for labels
		 *
		 * @return labels, a copy if a subset is set
		 */
		SGVector<float64_t> get_labels();

		/** get copy of labels.
		 *
		 * possible with subset
		 *
		 * @return labels
		 */
		SGVector<float64_t> get_labels_copy();

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

		/**
		 * set all labels to zero
		 *
		 * possible with subset
		 * */
		void zero();

		/**
		 * set all labels to a const value
		 *
		 * possible with subset
		 *
		 * @param c const to set labels to
		 * */
		void set_to_const(float64_t c);

		/** get INT label vector
		 *
		 * possible with subset
		 *
		 * @return INT labels
		 */
		SGVector<int32_t> get_int_labels();

		/** set INT labels
		 *
		 * not possible on subset
		 *
		 * @param labels INT labels
		 */
		void set_int_labels(SGVector<int32_t> labels);

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
		/** set INT64 labels
		 *
		 * not possible on subset
		 *
		 * @param labels INT labels
		 */
		void set_int_labels(SGVector<int64_t> labels);
#endif

		/** get number of labels, depending on whether a subset is set
		 *
		 * @return number of labels
		 */
		virtual int32_t get_num_labels() const;

		/** get label type
		 *
		 * @return label type (binary, multiclass, ...)
		 */
		virtual ELabelType get_label_type() const=0;

	public:
		/** label designates classify reject */
		static const int32_t REJECTION_LABEL = -2;

	private:
		void init();

	protected:
		/** the label vector */
		SGVector<float64_t> m_labels;
};
}
#endif
