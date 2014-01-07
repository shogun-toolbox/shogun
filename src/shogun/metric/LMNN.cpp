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
#ifdef HAVE_LAPACK

#include <metric/LMNN.h>
#include <metric/LMNNImpl.h>
#include <mathematics/Math.h>

/// useful shorthands to perform operations with Eigen matrices

// trace of the product of two matrices computed fast using trace(A*B)=sum(A.*B')
#define	TRACE(A,B)		(((A).array()*(B).transpose().array()).sum())

using namespace shogun;
using namespace Eigen;

CLMNN::CLMNN()
{
	init();

	m_statistics = new CLMNNStatistics();
	SG_REF(m_statistics);
}

CLMNN::CLMNN(CDenseFeatures<float64_t>* features, CMulticlassLabels* labels, int32_t k)
{
	init();

	m_features = features;
	m_labels = labels;
	m_k = k;

	SG_REF(m_features)
	SG_REF(m_labels)

	m_statistics = new CLMNNStatistics();
	SG_REF(m_statistics);
}

CLMNN::~CLMNN()
{
	SG_UNREF(m_features)
	SG_UNREF(m_labels)
	SG_UNREF(m_statistics);
}

const char* CLMNN::get_name() const
{
	return "LMNN";
}

void CLMNN::train(SGMatrix<float64_t> init_transform)
{
	SG_DEBUG("Entering CLMNN::train().\n")

	/// Check training data and arguments, initializing, if necessary, init_transform
	CLMNNImpl::check_training_setup(m_features, m_labels, init_transform);

	/// Initializations

	// cast is safe, check_training_setup ensures features are dense
	CDenseFeatures<float64_t>* x = static_cast<CDenseFeatures<float64_t>*>(m_features);
	CMulticlassLabels* y = CLabelsFactory::to_multiclass(m_labels);
	SG_DEBUG("%d input vectors with %d dimensions.\n", x->get_num_vectors(), x->get_num_features());

	// Use Eigen matrix for the linear transform L. The Mahalanobis distance is L^T*L
	MatrixXd L = Map<MatrixXd>(init_transform.matrix, init_transform.num_rows,
			init_transform.num_cols);
	// Compute target or genuine neighbours
	SG_DEBUG("Finding target nearest neighbors.\n")
	SGMatrix<index_t> target_nn = CLMNNImpl::find_target_nn(x, y, m_k);
	// Initialize (sub-)gradient
	SG_DEBUG("Summing outer products for (sub-)gradient initialization.\n")
	MatrixXd gradient = (1-m_regularization)*CLMNNImpl::sum_outer_products(x, target_nn);
	// Value of the objective function at every iteration
	SGVector<float64_t> obj(m_maxiter);
	// The step size is modified depending on how the objective changes, leave the
	// step size member unchanged and use a local one
	float64_t stepsize = m_stepsize;
	// Last active set of impostors computed exactly, current and previous impostors sets
	ImpostorsSetType exact_impostors, cur_impostors, prev_impostors;
	// Iteration counter
	uint32_t iter = 0;
	// Criterion for termination
	bool stop = false;
	// Make space for the training statistics
	m_statistics->resize(m_maxiter);

	/// Main loop
	while (!stop)
	{
		SG_PROGRESS(iter, 0, m_maxiter)

		// Find current set of impostors
		SG_DEBUG("Finding impostors.\n")
		cur_impostors = CLMNNImpl::find_impostors(x,y,L,target_nn,iter,m_correction);
		SG_DEBUG("Found %d impostors in the current set.\n", cur_impostors.size())

		// (Sub-) gradient computation
		SG_DEBUG("Updating gradient.\n")
		CLMNNImpl::update_gradient(x, gradient, cur_impostors, prev_impostors, m_regularization);
		// Take gradient step
		SG_DEBUG("Taking gradient step.\n")
		CLMNNImpl::gradient_step(L, gradient, stepsize, m_diagonal);

		// Compute the objective, trace of Mahalanobis distance matrix (L squared) times the gradient
		// plus the number of current impostors to account for the margin
		SG_DEBUG("Computing objective.\n")
		obj[iter] = TRACE(L.transpose()*L,gradient) + m_regularization*cur_impostors.size();

		// Correct step size
		CLMNNImpl::correct_stepsize(stepsize, obj, iter);

		// Check termination criterion
		stop = CLMNNImpl::check_termination(stepsize, obj, iter, m_maxiter, m_stepsize_threshold, m_obj_threshold);

		// Update iteration counter
		iter = iter + 1;
		// Update previous set of impostors
		prev_impostors = cur_impostors;

		// Store statistics for this iteration
		m_statistics->set(iter-1, obj[iter-1], stepsize, cur_impostors.size());

		SG_DEBUG("iteration=%d, objective=%.4f, #impostors=%4d, stepsize=%.4E\n",
				iter, obj[iter-1], cur_impostors.size(), stepsize)
	}

	// Truncate statistics in case convergence was reached in less than maxiter
	m_statistics->resize(iter);

	/// Store the transformation found in the class attribute
	int32_t nfeats = x->get_num_features();
	float64_t* cloned_data = SGMatrix<float64_t>::clone_matrix(L.data(), nfeats, nfeats);
	m_linear_transform = SGMatrix<float64_t>(cloned_data, nfeats, nfeats);

	SG_DEBUG("Leaving CLMNN::train().\n")
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
	REQUIRE(k>0, "The number of target neighbors per example must be larger than zero\n");
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
	REQUIRE(stepsize>0, "The step size used in gradient descent must be larger than zero\n")
	m_stepsize = stepsize;
}

float64_t CLMNN::get_stepsize_threshold() const
{
	return m_stepsize_threshold;
}

void CLMNN::set_stepsize_threshold(const float64_t stepsize_threshold)
{
	REQUIRE(stepsize_threshold>0,
			"The threshold for the step size must be larger than zero\n")
	m_stepsize_threshold = stepsize_threshold;
}

uint32_t CLMNN::get_maxiter() const
{
	return m_maxiter;
}

void CLMNN::set_maxiter(const uint32_t maxiter)
{
	REQUIRE(maxiter>0, "The number of maximum iterations must be larger than zero\n")
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

float64_t CLMNN::get_obj_threshold() const
{
	return m_obj_threshold;
}

void CLMNN::set_obj_threshold(const float64_t obj_threshold)
{
	REQUIRE(obj_threshold>0,
			"The threshold for the objective must be larger than zero\n")
	m_obj_threshold = obj_threshold;
}

bool CLMNN::get_diagonal() const
{
	return m_diagonal;
}

void CLMNN::set_diagonal(const bool diagonal)
{
	m_diagonal = diagonal;
}

CLMNNStatistics* CLMNN::get_statistics() const
{
	SG_REF(m_statistics);
	return m_statistics;
}

void CLMNN::init()
{
	SG_ADD(&m_linear_transform, "linear_transform",
			"Linear transform in matrix form", MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_features, "features", "Training features",
			MS_NOT_AVAILABLE)
	SG_ADD((CSGObject**) &m_labels, "labels", "Training labels",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_k, "k", "Number of target neighbours per example",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_regularization, "regularization", "Regularization",
			MS_AVAILABLE)
	SG_ADD(&m_stepsize, "stepsize", "Step size in gradient descent",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_stepsize_threshold, "stepsize_threshold", "Step size threshold",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_maxiter, "maxiter", "Maximum number of iterations",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_correction, "correction",
			"Iterations between exact impostors search", MS_NOT_AVAILABLE)
	SG_ADD(&m_obj_threshold, "obj_threshold", "Objective threshold",
			MS_NOT_AVAILABLE)
	SG_ADD(&m_diagonal, "m_diagonal", "Diagonal transformation", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_statistics, "statistics", "Training statistics",
			MS_NOT_AVAILABLE);

	m_features = NULL;
	m_labels = NULL;
	m_k = 1;
	m_regularization = 0.5;
	m_stepsize = 1e-07;
	m_stepsize_threshold = 1e-22;
	m_maxiter = 1000;
	m_correction = 15;
	m_obj_threshold = 1e-9;
	m_diagonal = false;
	m_statistics = NULL;
}

CLMNNStatistics::CLMNNStatistics()
{
	init();
}

CLMNNStatistics::~CLMNNStatistics()
{
}

const char* CLMNNStatistics::get_name() const
{
	return "LMNNStatistics";
}

void CLMNNStatistics::resize(int32_t size)
{
	REQUIRE(size > 0, "The new size in CLMNNStatistics::resize must be larger than zero."
			 " Given value is %d.\n", size);

	obj.resize_vector(size);
	stepsize.resize_vector(size);
	num_impostors.resize_vector(size);
}

void CLMNNStatistics::set(index_t iter, float64_t obj_iter, float64_t stepsize_iter,
		uint32_t num_impostors_iter)
{
	REQUIRE(iter >= 0 && iter < obj.vlen, "The iteration index in CLMNNStatistics::set "
			"must be larger or equal to zero and less than the size (%d). Given valu is %d.\n", obj.vlen, iter);

	obj[iter] = obj_iter;
	stepsize[iter] = stepsize_iter;
	num_impostors[iter] = num_impostors_iter;
}

void CLMNNStatistics::init()
{
	SG_ADD(&obj, "obj", "Objective at each iteration", MS_NOT_AVAILABLE);
	SG_ADD(&stepsize, "stepsize", "Step size at each iteration", MS_NOT_AVAILABLE);
	SG_ADD(&num_impostors, "num_impostors", "Number of impostors at each iteration",
			MS_NOT_AVAILABLE);
}

#endif /* HAVE_LAPACK */
#endif /* HAVE_EIGEN3 */
