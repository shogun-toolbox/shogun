/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifdef HAVE_EIGEN3

#include <shogun/metric/LMNN.h>
#include <shogun/metric/LMNNImpl.h>
#include <iostream>

using namespace shogun;
using namespace Eigen;

CLMNN::CLMNN()
{
	init();
}

CLMNN::CLMNN(CDenseFeatures<float64_t>* features, CMulticlassLabels* labels, int32_t k)
{
	init();

	m_features = features;
	m_labels = labels;
	m_k = k;

	SG_REF(m_features)
	SG_REF(m_labels)
}

CLMNN::~CLMNN()
{
	SG_UNREF(m_features)
	SG_UNREF(m_labels)
}

const char* CLMNN::get_name() const
{
	return "LMNN";
}

void CLMNN::train(SGMatrix<float64_t> init_transform)
{
	SG_DEBUG("Entering CLMNN::train().\n")

	/// Check training data and the argument

	REQUIRE(m_features->has_property(FP_DOT),
			"LMNN can only be applied to features that support dot products\n")
	REQUIRE(m_labels->get_label_type()==LT_MULTICLASS,
			"LMNN supports only MulticlassLabels\n")
	REQUIRE(m_labels->get_num_labels()==m_features->get_num_vectors(),
			"The number of feature vectors must be equal to the number of labels\n")
	//FIXME this requirement should be dropped in the future
	REQUIRE(m_features->get_feature_class()==C_DENSE,
			"Currently, LMNN supports only DenseFeatures\n")

	CDenseFeatures<float64_t>* x = dynamic_cast<CDenseFeatures<float64_t>*>(m_features);
	CMulticlassLabels* y = CLabelsFactory::to_multiclass(m_labels);

	REQUIRE(init_transform.num_rows==x->get_num_features() &&
			init_transform.num_rows==init_transform.num_cols,
			"The initial transform must be a square matrix of size equal to the "
			"number of features\n")

	/// Initializations

	// Use Eigen matrix for the linear transform L. The Mahalanobis distance is L^T*L
	MatrixXd L = Map<MatrixXd>(init_transform.matrix, init_transform.num_rows,
			init_transform.num_cols);
	// Compute target or genuine neighbours
	SG_DEBUG("Finding target nearest neighbors.\n")
	SGMatrix<index_t> target_nn = CLMNNImpl::find_target_nn(x, y, m_k);
	// Compute outer products between all pairs of feature vectors
	SG_DEBUG("Computing outer products.\n")
	OuterProductsMatrixType outer_products = CLMNNImpl::compute_outer_products(x);
	// Initialize (sub-)gradient
	SG_DEBUG("Summing outer products for (sub-)gradient initilization.\n")
	MatrixXd gradient = (1-m_regularization)*CLMNNImpl::sum_outer_products(outer_products, target_nn);
	// Stop criterion
	bool stop = false;
	// Value of the objective function at every iteration
	SGVector<float64_t> obj(m_maxiter);
	// The step size is modified depending on how the objective changes, leave the
	// step size member unchanged and use a local one
	float64_t stepsize = m_stepsize;
	// Last active set of impostors computed exactly, current and previous impostors sets
	ImpostorsSetType exact_impostors, cur_impostors, prev_impostors;
	// Iteration counter
	uint32_t iter = 0;

	/// Main loop
	while (not stop && iter < m_maxiter)
	{
		SG_PROGRESS(iter, 0, m_maxiter)

		// Find current set of impostors
		SG_DEBUG("Finding impostors.\n")
		if ((iter % m_correction)==0)
		{
			// exact computation of impostors
			exact_impostors = CLMNNImpl::find_impostors(ExactSearch, x, y, L, target_nn, exact_impostors);
			cur_impostors = exact_impostors;
		}
		else
		{
			cur_impostors = CLMNNImpl::find_impostors(ApproxSearch, x, y, L, target_nn, exact_impostors);
		}
		SG_DEBUG("Found %d impostors in the current set.\n", cur_impostors.size())

		// (Sub-) gradient computation
		SG_DEBUG("Updating gradient.\n")
		CLMNNImpl::update_gradient(gradient, outer_products, cur_impostors,
				prev_impostors, m_regularization);
		// Take gradient step
		SG_DEBUG("Taking gradient step.\n")
		CLMNNImpl::gradient_step(L, gradient, stepsize);

		// Compute objective
		SG_DEBUG("Computing objective.\n")
		obj[iter] = CLMNNImpl::compute_objective(x, L, outer_products, target_nn,
				cur_impostors, m_regularization);

		// Correct step size
		if (iter > 0)
		{
			// Difference between current and previous objective
			float64_t delta = obj[iter] - obj[iter-1];

			if (delta > 0)
			{
				// The objective has increased, we have probably jumped over the optimum,
				// thus, decrease the step size
				stepsize *= 0.5;
			}
			else
			{
				// The objective has decreased, we are in the right direction,
				// increase the step size
				stepsize *= 1.01;
			}
		}

		// Update iteration counter
		iter = iter + 1;
		// Update previous set of impostors
		prev_impostors = cur_impostors;

		SG_DEBUG("iteration=%d, objective=%.4f, stepsize=%.4E\n", iter, obj[iter], stepsize)
	}

	/// Store the transformation found in the class attribute

	int32_t nfeats = x->get_num_features();
	float64_t* cloned_data = SGMatrix<float64_t>::clone_matrix(L.data(), nfeats, nfeats);
	m_linear_transform = SGMatrix<float64_t>(cloned_data, nfeats, nfeats);
}

SGMatrix<float64_t> CLMNN::get_linear_transform() const
{
	return m_linear_transform;
}

CCustomMahalanobisDistance* CLMNN::get_distance() const
{
	// Compute Mahalanobis distance matrix M = L^T*L

	// Put the linear transform L in Eigen to perform the matrix multiplication
	// L is not copied to another region of memory
	Map<const MatrixXd> map_linear_transform(m_linear_transform.matrix,
			m_linear_transform.num_rows, m_linear_transform.num_cols);
	// TODO exploit that M is symmetric
	MatrixXd M = map_linear_transform.transpose()*map_linear_transform;
	// TODO avoid copying
	SGMatrix<float64_t> mahalanobis_matrix(M.rows(), M.cols());
	for (index_t i = 0; i < M.rows(); i++)
		for (index_t j = 0; j < M.cols(); j++)
			mahalanobis_matrix(i,j) = M(i,j);

	// Create custom Mahalanobis distance with matrix M associated with the training features

	CCustomMahalanobisDistance* distance =
			new CCustomMahalanobisDistance(m_features, m_features, mahalanobis_matrix);
	SG_REF(distance)

	return distance;
}

int32_t CLMNN::get_k() const
{
	return m_k;
}

void CLMNN::set_k(const int32_t k)
{
	REQUIRE(k>0, "The number of target neighbors per example must be greater than zero\n");
	m_k = k;
}

float64_t CLMNN::get_regularization() const
{
	return m_regularization;
}

void CLMNN::set_regularization(const float64_t regularization)
{
	m_regularization = regularization;
}

float64_t CLMNN::get_stepsize() const
{
	return m_stepsize;
}

void CLMNN::set_stepsize(const float64_t stepsize)
{
	m_stepsize = stepsize;
}

uint32_t CLMNN::get_maxiter() const
{
	return m_maxiter;
}

void CLMNN::set_maxiter(const uint32_t maxiter)
{
	m_maxiter = maxiter;
}

uint32_t CLMNN::get_correction() const
{
	return m_correction;
}

void CLMNN::set_correction(const uint32_t correction)
{
	m_correction = correction;
}

void CLMNN::init()
{
	SG_ADD(&m_linear_transform, "m_linear_transform",
			"Linear transform in matrix form", MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_features, "m_features", "Training features",
			MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_labels, "m_labels", "Training labels",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_k, "m_k", "Number of target neighbours per example",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_regularization, "m_regularization", "Regularization",
			MS_AVAILABLE)
	SG_ADD(&m_stepsize, "m_stepsize", "Step size in gradient descent",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_maxiter, "m_maxiter", "Maximum number of iterations",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_correction, "m_correction",
			"Iterations between exact impostors search", MS_NOT_AVAILABLE)

	m_features = NULL;
	m_labels = NULL;
	m_k = 1;
	m_regularization = 0.5;
	m_stepsize = 1e-07;
	m_maxiter = 1000;
	m_correction = 15;
}

#endif /* HAVE_EIGEN3 */
