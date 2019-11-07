/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Saurabh Mahindre, Olivier NGuyen, Thoralf Klein, Giovanni De Toni,
 *          Heiko Strathmann, Michele Mazzoni, Evgeniy Andreev, Yuyu Zhang,
 *          Chiyuan Zhang, Viktor Gal, Bjoern Esser
 */

#ifndef _MULTICLASS_LABELS__H__
#define _MULTICLASS_LABELS__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/File.h>
#include <shogun/labels/LabelTypes.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/features/SubsetStack.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
	class File;
	class BinaryLabels;
	class MulticlassLabels;
	class DenseLabels;

/** @brief Multiclass Labels for multi-class classification
 *
 * valid values for labels are 0...nr_classes-1
 */
class MulticlassLabels : public DenseLabels
{
	public:
		/** default constructor */
		MulticlassLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		MulticlassLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 */
		MulticlassLabels(SGVector<float64_t> src);


		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		MulticlassLabels(std::shared_ptr<File> loader);

		/**
		 * Convert binary labels to multiclass labels,
		 * namely -1 is mapped to 0 and 1 to 1.
		 *
		 * @param labels Binary labels
		 */
		MulticlassLabels(const std::shared_ptr<BinaryLabels>& labels);

		/** copy constructor */
		MulticlassLabels(const MulticlassLabels& orig);

		/** destructor */
		~MulticlassLabels();

		/** Make sure the label is valid, otherwise raise SG_ERROR.
		 *
		 * possible with subset
		 *
		 * @param context optional message to convey the context
		 */
		virtual void ensure_valid(const char* context=NULL);

		bool is_valid() const;

		/** get label type
		 *
		 * @return label type multiclass
		 */
		virtual ELabelType get_label_type() const;

		virtual std::shared_ptr<Labels> duplicate() const;

		/** returns labels containing +1 at positions with ith class
		 *  and -1 at other positions
		 *  @param i index of class
		 *  @return new binary labels
		 */
		std::shared_ptr<BinaryLabels> get_binary_for_class(int32_t i);

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

		/** Returns confidence scores for given class
		* @param i Class index
		* @return Confidences of class i
		*/
		SGVector<float64_t> get_confidences_for_class(int32_t i);

		/** @return object name */
		virtual const char* get_name() const { return "MulticlassLabels"; }
#ifndef SWIG // SWIG should skip this part
		virtual std::shared_ptr<Labels> shallow_subset_copy();
#endif
		/**
		 * Cast a generic label object to a multiclass one
		 * @param labels generic Labels instance
		 * @return the casted pointer (already SG_REF'ed)
		 */
		static std::shared_ptr<MulticlassLabels> obtain_from_generic(const std::shared_ptr<Labels>& labels);

	private:
		/** initialises and register parameters */
		void init();

	protected:

		/** multiclass confidences */
		SGMatrix<float64_t> m_multiclass_confidences;
};

#ifndef SWIG
std::shared_ptr<MulticlassLabels> multiclass_labels(const std::shared_ptr<Labels>& orig);
#endif // SWIG
}
#endif
