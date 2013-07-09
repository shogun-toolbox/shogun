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
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>

using namespace shogun;
using namespace Eigen;

CLMNN::CLMNN()
{
	init();
}

CLMNN::CLMNN(CFeatures* features, CMulticlassLabels* labels, uint32_t k)
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
	/// Check training data
	REQUIRE(m_labels->get_num_labels()==m_features->get_num_vectors(),
			"The number of feature vectors must be equal to the number of labels\n")

	/// Initializations

	// Linear transform L, the Mahalanobis distance is L^T*L
	MatrixXd L = Map<MatrixXd>(init_transform.matrix, init_transform.num_rows,
			init_transform.num_cols);
	// Iteration counter
	uint32_t iter = 0;
	// Previous active set of impostors, empty at first
	ImpostorsSetType prev_impostors;
	// Compute target or genuine neighbours
	SGMatrix<int32_t> target_nn = find_target_nn();
	// Compute outer products between all pairs of feature vectors
	OuterProductsMatrixType outer_products = compute_outer_products();
	// Initialize (sub-)gradient
	MatrixXd gradient = sum_outer_products(outer_products, target_nn);
	// Stop criterion
	bool stop = false;
	// Value of the objective function at every iteration
	SGVector<float64_t> obj(m_maxiter);
	// The step size is modified depending on how the objective changes, leave the
	// step size member unchanged and use a local one
	float64_t stepsize = m_stepsize;

	/// Main loop
	while (not stop && iter < m_maxiter)
	{
		SG_PROGRESS(iter, 0, m_maxiter)

		//FIXME add approximate computation of impostors set
		// Find current set of impostors
		ImpostorsSetType cur_impostors = find_impostors(L, target_nn);

		// (Sub-) gradient computation
		update_gradient(gradient, outer_products, cur_impostors, prev_impostors);
		// Take gradient step and project onto the positive semi-definite cone
		gradient_step(L, gradient, stepsize);

		// Compute objective
		obj[iter] = compute_objective(L, outer_products, target_nn, cur_impostors);

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
	}
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

void CLMNN::init()
{
	SG_ADD(&m_linear_transform, "m_linear_transform", "Linear transform in matrix form",
			MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_features, "m_features", "Training features", MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_labels, "m_labels", "Training labels", MS_NOT_AVAILABLE)
	SG_ADD(&m_k, "m_k", "Number of target neighbours per example", MS_NOT_AVAILABLE)
	SG_ADD(&m_regularization, "m_regularization", "Regularization", MS_AVAILABLE)
	SG_ADD(&m_stepsize, "m_stepsize", "Step size in gradient descent", MS_NOT_AVAILABLE)
	SG_ADD(&m_maxiter, "m_maxiter", "Maximum number of iterations", MS_NOT_AVAILABLE)

	m_features = NULL;
	m_labels = NULL;
	m_k = 1;
	m_regularization = 0.5;
	m_stepsize = 1e-07;
	m_maxiter = 1000;
}

//TODO
SGMatrix<int32_t> CLMNN::find_target_nn() const
{
	return SGMatrix<int32_t>(m_k, m_features->get_num_vectors());
}

//TODO
OuterProductsMatrixType CLMNN::compute_outer_products() const
{
	return OuterProductsMatrixType();
}

//TODO
MatrixXd CLMNN::sum_outer_products(const OuterProductsMatrixType& C, const SGMatrix<int32_t> idxs) const
{
	return VectorXd();
}

//TODO
ImpostorsSetType CLMNN::find_impostors(const MatrixXd& L, const SGMatrix<int32_t> target_nn) const
{
	return ImpostorsSetType();
}

//TODO
void CLMNN::update_gradient(MatrixXd& G, const OuterProductsMatrixType& C, const ImpostorsSetType& Nc, const ImpostorsSetType& Np) const
{
}

//TODO
void CLMNN::gradient_step(MatrixXd& L, const MatrixXd& G, float64_t stepsize) const
{
}

//TODO
float64_t CLMNN::compute_objective(const MatrixXd& L, const OuterProductsMatrixType& C, const SGMatrix<int32_t> target_nn, const ImpostorsSetType& Nc) const
{
	return 0.0;
}

#endif /* HAVE_EIGEN3 */
