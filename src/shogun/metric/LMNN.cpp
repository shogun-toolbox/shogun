/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Heiko Strathmann, Giovanni De Toni, Viktor Gal,
 * Wuwei Lin
 */

#include <shogun/metric/LMNN.h>

#include <shogun/base/progress.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/metric/LMNNImpl.h>

using namespace shogun;

CLMNN::CLMNN()
{
	init();

	m_statistics = new CLMNNStatistics();
	SG_REF(m_statistics);
}

CLMNN::CLMNN(CFeatures* features, CMulticlassLabels* labels, int32_t k)
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

	// Check training data and arguments, initializing, if necessary, init_transform
	CLMNNImpl::check_training_setup(m_features, m_labels, init_transform, m_k);

	// Initializations

	// cast is safe, check_training_setup ensures features are dense
	CDenseFeatures<float64_t>* x = static_cast<CDenseFeatures<float64_t>*>(m_features);
	CMulticlassLabels* y = multiclass_labels(m_labels);
	SG_DEBUG("%d input vectors with %d dimensions.\n", x->get_num_vectors(), x->get_num_features());

	auto& L = init_transform;
	// Compute target or genuine neighbours
	SG_DEBUG("Finding target nearest neighbors.\n")
	SGMatrix<index_t> target_nn = CLMNNImpl::find_target_nn(x, y, m_k);
	// Initialize (sub-)gradient
	SG_DEBUG("Summing outer products for (sub-)gradient initialization.\n")
	auto gradient = CLMNNImpl::sum_outer_products(x, target_nn);
	linalg::scale(gradient, gradient, 1 - m_regularization);
	// Value of the objective function at every iteration
	SGVector<float64_t> obj(m_maxiter);
	// The step size is modified depending on how the objective changes, leave the
	// step size member unchanged and use a local one
	float64_t stepsize = m_stepsize;
	// Last active set of impostors computed exactly, current and previous impostors sets
	ImpostorsSetType exact_impostors, cur_impostors, prev_impostors;
	// Iteration counter
	int32_t iter = 0;
	// Criterion for termination
	bool stop = false;
	// Make space for the training statistics
	m_statistics->resize(m_maxiter);

	// Progress bar
	auto pb = SG_PROGRESS(range(m_maxiter));

	// Main loop
	while (!stop)
	{
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
		obj[iter] = m_regularization * cur_impostors.size();
		obj[iter] +=
		    linalg::trace_dot(linalg::matrix_prod(L, L, true, false), gradient);

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

		// Print progress bar iteration
		pb.print_progress();
	}
	pb.complete();

	// Truncate statistics in case convergence was reached in less than maxiter
	m_statistics->resize(iter);

	// Store the transformation found in the class attribute
	int32_t nfeats = x->get_num_features();
	float64_t* cloned_data =
	    SGMatrix<float64_t>::clone_matrix(L.matrix, nfeats, nfeats);
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
	auto M = linalg::matrix_prod(
	    m_linear_transform, m_linear_transform, true, false);

	// Create custom Mahalanobis distance with matrix M associated with the training features
	auto distance = new CCustomMahalanobisDistance(m_features, m_features, M);
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

int32_t CLMNN::get_maxiter() const
{
	return m_maxiter;
}

void CLMNN::set_maxiter(const int32_t maxiter)
{
	REQUIRE(maxiter>0, "The number of maximum iterations must be larger than zero\n")
	m_maxiter = maxiter;
}

int32_t CLMNN::get_correction() const
{
	return m_correction;
}

void CLMNN::set_correction(const int32_t correction)
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
			"Linear transform in matrix form");
	SG_ADD((CSGObject**) &m_features, "features", "Training features");
	SG_ADD((CSGObject**) &m_labels, "labels", "Training labels");
	SG_ADD(&m_k, "k", "Number of target neighbours per example");
	SG_ADD(&m_regularization, "regularization", "Regularization",
			ParameterProperties::HYPER);
	SG_ADD(&m_stepsize, "stepsize", "Step size in gradient descent");
	SG_ADD(&m_stepsize_threshold, "stepsize_threshold", "Step size threshold");
	SG_ADD(&m_maxiter, "maxiter", "Maximum number of iterations");
	SG_ADD(&m_correction, "correction",
			"Iterations between exact impostors search");
	SG_ADD(&m_obj_threshold, "obj_threshold", "Objective threshold");
	SG_ADD(&m_diagonal, "m_diagonal", "Diagonal transformation");
	SG_ADD((CSGObject**) &m_statistics, "statistics", "Training statistics");

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
	SG_ADD(&obj, "obj", "Objective at each iteration");
	SG_ADD(&stepsize, "stepsize", "Step size at each iteration");
	SG_ADD(&num_impostors, "num_impostors", "Number of impostors at each iteration");
}

