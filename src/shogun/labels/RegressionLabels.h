/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Soeren Sonnenburg, Evgeniy Andreev, Yuyu Zhang, 
 *          Chiyuan Zhang, Fernando Iglesias, Sergey Lisitsyn
 */

#ifndef _REAL_LABELS__H__
#define _REAL_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/features/SubsetStack.h>

namespace shogun
{
	class CFile;
	class CRegressionLabels;
	class CDenseLabels;

/** @brief Real Labels are real-valued labels
 *
 * They are used for regression problems and as outputs of classifiers.
 *
 * valid values for labels are all real-valued numbers.
 */
class CRegressionLabels : public CDenseLabels
{
	public:
		/** default constructor */
		CRegressionLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CRegressionLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 */
		CRegressionLabels(const SGVector<float64_t> src);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CRegressionLabels(CFile* loader);

		/** get label type
		 *
		 * @return label type real
		 */
		virtual ELabelType get_label_type() const;

		/** @return object name */
		virtual const char* get_name() const { return "RegressionLabels"; }

#ifndef SWIG // SWIG should skip this part
		virtual CLabels* shallow_subset_copy();
#endif
};
}
#endif
