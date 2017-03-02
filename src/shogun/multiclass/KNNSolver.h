/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef KNNSOLVER_H__
#define KNNSOLVER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
/* Virtual base class for all KNN solvers */

class CDistanceMachine;
class CKNNSolver : public CDistanceMachine
{
	public:
		/** default constructor */
		CKNNSolver(): CDistanceMachine() { init(); }

		/** deconstructor */
		virtual ~CKNNSolver() { /* nothing to do */ }

		/** constructor
		 *
		 * @param k k
		 * @param q m_q
		 * @param num_classes m_num_classes
		 * @param min_label m_min_label
		 * @param train_labels m_train_labels 
		 */
		CKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels);

		/** compute the histogram of class outputs of the k nearest 
		 * neighbors to a test vector and return the index of the most frequent class
		 *
		 * @param classes vector used to store the histogram
		 * @param train_lab class indices of the training data. If the cover
		 * tree is not used, the elements are ordered by increasing distance
		 * and there are elements for each of the training vectors. If the cover
		 * tree is used, it contains just m_k elements not necessary ordered.+		 *
		 * @return index of the most frequent class, class detected by KNN
		 */
		int32_t choose_class(float64_t* classes, const int32_t* train_lab) const;

		/** compute the histogram of class outputs of the k nearest neighbors 
		 *  to a test vector, using k from 1 to m_k, and write the most frequent
		 *  class for each value of k in output, using a distance equal to step
		 *  between elements in the output array
		 *
		 * @param output return value where the most frequent classes are written
		 * @param classes vector used to store the histogram
		 * @param train_lab class indices of the training data; no matter the cover tree
		 * is used or not, the neighbors are ordered by distance to the test vector
		 * in ascending order
		 * @param step distance between elements to be written in output
		 */
		void choose_class_for_multiple_k(int32_t* output, int32_t* classes, const int32_t* train_lab, const int32_t step) const;

		/** classify objects, and the implementation will depended on which knn solver been choosen.
		 *
		 * @param d distance
		 * @param num_lab number of labels been used to classify 
		 * @param train_lab class indices of the training data
		 * @param classes vector used to store the histogram
		 * @return the classified labels
		 */
		 virtual CMulticlassLabels* classify_objects(CDistance* d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const = 0;

		/**
		 * classify all objects, and the implementation will depended on which knn solver been choosen.
		 * @param d distance
		 * @param num_lab number of labels been used to classify 
		 * @param train_lab class indices of the training data
		 * @param classes vector used to store the histogram
		 * @return the classified labels
		 */
		 virtual SGVector<int32_t> classify_objects_k(CDistance* d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes) const = 0;

		/** @return object name */
		virtual const char* get_name() const { return "KNNSolver"; }

	private:
		void init();

	protected:
		/// the k parameter in KNN
		int32_t m_k;

		/// parameter q of rank weighting
		float64_t m_q;

		/// number of classes (i.e. number of values labels can take)
		int32_t m_num_classes;

		/// smallest label, i.e. -1
		int32_t m_min_label;
 
		/** the actual trainlabels */
		SGVector<int32_t> m_train_labels;
};

}
#endif
