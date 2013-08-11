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

#ifndef _MULTICLASS_LABELS__H__
#define _MULTICLASS_LABELS__H__

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/features/SubsetStack.h>

namespace shogun
{
	class CFile;
	class CBinaryLabels;
	class CMulticlassLabels;
	class CDenseLabels;

/** @brief Multiclass Labels for multi-class classification 
 *
 * valid values for labels are 0...nr_classes-1
 */
class CMulticlassLabels : public CDenseLabels
{
	public:
		/** default constructor */
		CMulticlassLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CMulticlassLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 */
		CMulticlassLabels(SGVector<float64_t> src);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CMulticlassLabels(CFile* loader);

		/** destructor */
		~CMulticlassLabels();

		/** Make sure the label is valid, otherwise raise SG_ERROR.
		 *
		 * possible with subset
         *
         * @param context optional message to convey the context
		 */
		virtual void ensure_valid(const char* context=NULL);

		/** get label type
		 *
		 * @return label type multiclass
		 */
		virtual ELabelType get_label_type() const;

		/** returns labels containing +1 at positions with ith class
		 *  and -1 at other positions
		 *  @param i index of class
		 *  @return new binary labels
		 */
		CBinaryLabels* get_binary_for_class(int32_t i);

		/** get unique labels (new SGVector)
		 *
		 * possible with subset
		 *
		 * @return unique labels
		 */
		SGVector<float64_t> get_unique_labels();

		/** return number of classes (for multiclass)
		 *
		 * possible with subset
		 *
		 * @return number of classes
		 */
		int32_t get_num_classes();

		/** returns multiclass confidences
		 *
		 * @param i index
		 * @return confidences of ith result
		 */
		SGVector<float64_t> get_multiclass_confidences(int32_t i);

		/** sets multiclass confidences. to prepare labels
		 * for storing multiclass confidences call
		 * @ref allocate_confidences_for
		 *
		 * @param i index
		 * @param confidences confidences to be set for ith result
		 */
		void set_multiclass_confidences(int32_t i, SGVector<float64_t> confidences);

		/** allocates matrix to store confidences. should always 
		 * be called before setting confidences with 
		 * @ref set_multiclass_confidences
		 *
		 * @param n_classes number of classes
		 */
		void allocate_confidences_for(int32_t n_classes);

		/** @return object name */
		virtual const char* get_name() const { return "MulticlassLabels"; }

	private:
		/** initialises and register parameters */
		void init();

	protected:

		/** multiclass confidences */
		SGMatrix<float64_t> m_multiclass_confidences;
};
}
#endif
