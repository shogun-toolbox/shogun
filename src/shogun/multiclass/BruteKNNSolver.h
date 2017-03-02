/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef BRUTESOLVER_H__
#define BRUTESOLVER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/multiclass/KNNSolver.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

/* Standard KNN solver. Test points are compared to all training data for each prediction. */
class CBruteKNNSolver : public CKNNSolver
{
	public:
		/** default constructor */
		CBruteKNNSolver() : CKNNSolver()
		{
			init(); 
		}

		/** deconstructor */
		virtual ~CBruteKNNSolver() { /* nothing to do */ }

		/** constructor
		 *
		 * @param k k
		 * @param q m_q
		 * @param num_classes m_num_classes
		 * @param min_label m_min_label
		 * @param train_labels m_train_labels
		 * @param NN nn
		 */
		CBruteKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels, const SGMatrix<index_t> NN);

		virtual CMulticlassLabels* classify_objects(CDistance* d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const;

		virtual SGVector<int32_t> classify_objects_k(CDistance* d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes) const;

		/** @return object name */
		const char* get_name() const { return "BruteKNNSolver"; }

	private:
		void init()
		{
			nn=SGMatrix<index_t>(3, 0);
		}

	protected:
		/** The nearest neighbors martix */
		SGMatrix<index_t> nn;

};

}
#endif
