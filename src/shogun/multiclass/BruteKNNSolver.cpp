/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/multiclass/BruteKNNSolver.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CBruteKNNSolver::CBruteKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels, const SGMatrix<index_t> NN):
CKNNSolver(k, q, num_classes, min_label, train_labels)
{
	init();
	nn=NN;
}

CMulticlassLabels* CBruteKNNSolver::classify_objects(CDistance* distance, const int32_t num_lab, int32_t* train_lab, float64_t* classes) const
{
	CMulticlassLabels* output=new CMulticlassLabels(num_lab);
	//get the k nearest neighbors of each example
	SGMatrix<index_t> NN = this->nn;

	//from the indices to the nearest neighbors, compute the class labels
	for (index_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
			train_lab[j] = m_train_labels[ NN(j,i) ];

		//get the index of the 'nearest' class
		index_t out_idx = choose_class(classes, train_lab);
		//write the label of 'nearest' in the output
		output->set_label(i, out_idx + m_min_label);
	}

	return output;
}

int32_t* CBruteKNNSolver::classify_objects_k(CDistance* distance, const int32_t num_lab, int32_t* train_lab, int32_t* classes) const
{
	int32_t* output=SG_MALLOC(int32_t, m_k*num_lab);

	//get the k nearest neighbors of each example
	SGMatrix<index_t> NN = this->nn;

	for (index_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
			train_lab[j] = m_train_labels[ NN(j,i) ];

		choose_class_for_multiple_k(output+i, classes, train_lab, num_lab);
	}

	return output;
}
