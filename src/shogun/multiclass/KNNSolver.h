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
		CKNNSolver(
		    const index_t k, const float64_t q, const index_t num_classes,
		    const index_t min_label, const SGVector<index_t> train_labels);

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
		index_t
		choose_class(float64_t* classes, const index_t* train_lab) const;

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
		void choose_class_for_multiple_k(
		    index_t* output, index_t* classes, const index_t* train_lab,
		    const index_t step) const;

		/** classify objects, and the implementation will depended on which knn solver been choosen.
		 *
		 * @param d distance
		 * @param num_lab number of labels been used to classify 
		 * @param train_lab class indices of the training data
		 * @param classes vector used to store the histogram
		 * @return the classified labels
		 */
		virtual CMulticlassLabels* classify_objects(
		    CDistance* d, const index_t num_lab, SGVector<index_t>& train_lab,
		    SGVector<float64_t>& classes) const = 0;

		/**
		 * classify all objects, and the implementation will depended on which knn solver been choosen.
		 * @param d distance
		 * @param num_lab number of labels been used to classify 
		 * @param train_lab class indices of the training data
		 * @param classes vector used to store the histogram
		 * @return the classified labels
		 */
		virtual SGVector<index_t> classify_objects_k(
		    CDistance* d, const index_t num_lab, SGVector<index_t>& train_lab,
		    SGVector<index_t>& classes) const = 0;

		/** @return object name */
		virtual const char* get_name() const { return "KNNSolver"; }

	private:
		void init();

	protected:
		/// the k parameter in KNN
		index_t m_k;

		/// parameter q of rank weighting
		float64_t m_q;

		/// number of classes (i.e. number of values labels can take)
		index_t m_num_classes;

		/// smallest label, i.e. -1
		index_t m_min_label;

		/** the actual trainlabels */
		SGVector<index_t> m_train_labels;
};

}
#endif
