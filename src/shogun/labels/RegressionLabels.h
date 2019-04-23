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
	class File;
	class RegressionLabels;
	class DenseLabels;

/** @brief Real Labels are real-valued labels
 *
 * They are used for regression problems and as outputs of classifiers.
 *
 * valid values for labels are all real-valued numbers.
 */
class RegressionLabels : public DenseLabels
{
	public:
		/** default constructor */
		RegressionLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		RegressionLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 */
		RegressionLabels(const SGVector<float64_t> src);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		RegressionLabels(std::shared_ptr<File> loader);

		/** get label type
		 *
		 * @return label type real
		 */
		virtual ELabelType get_label_type() const;

		/** @return object name */
		virtual const char* get_name() const { return "RegressionLabels"; }

		/** shallow-copy of the labels object
		 * @see Labels::duplicate
		 */
		virtual std::shared_ptr<Labels> duplicate() const;

#ifndef SWIG // SWIG should skip this part
		virtual std::shared_ptr<Labels> shallow_subset_copy();
#endif
};

#ifndef SWIG
std::shared_ptr<RegressionLabels> regression_labels(std::shared_ptr<Labels> orig);
#endif // SWIG
}
#endif
