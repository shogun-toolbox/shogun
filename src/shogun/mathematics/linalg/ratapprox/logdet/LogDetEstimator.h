/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Heiko Strathmann, Bjoern Esser,
 *          Viktor Gal
 */

#ifndef LOG_DET_ESTIMATOR_H_
#define LOG_DET_ESTIMATOR_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
class TraceSampler;
template<class T> class OperatorFunction;
template<class T> class SGVector;
template<class T> class SGMatrix;

/** @brief Class to create unbiased estimators of \f$log(\left|C\right|)=
 * trace(log(C))\f$. For each estimate, it samples trace vectors (one by one)
 * and calls solve of OperatorFunction, stores the resulting in
 * a vector, then simply computes running averages over the estimates
 */
class LogDetEstimator : public SGObject
{
public:
	/** Default constructor */
	LogDetEstimator();

#ifdef HAVE_LAPACK
	/**
	 * Convenient constructor for SGSparseMatrix<float64_t>. Works only when
	 * Eigen3 and LAPACK libraries are available.
	 *
	 * Uses the default configuration:
	 * - LanczosEigenSolver,
	 * - CLogRationalApproximationCGM with 1E-5 accuracy
	 * - CGMShiftedFamilySolver,
	 * when COLPACK package is available then CProbingsampler with power = 1,
	 * EOrderingVariant NATURAL, EColoringVariant DISTANCE_TWO is used,
	 * Otherwise CNormalsampler is used.
	 *
	 * @param sparse_mat the input sparse matrix
	 */
	LogDetEstimator(SGSparseMatrix<float64_t> sparse_mat);
#endif

	/**
	 * Constructor
	 *
	 * @param trace_sampler the trace sampler
	 * @param operator_log the operator function
	 */
	LogDetEstimator(
		std::shared_ptr<TraceSampler> trace_sampler,
		std::shared_ptr<OperatorFunction<float64_t>> operator_log);

	/** Destructor */
	virtual ~LogDetEstimator();

	/**
	 * Method that gives num_estimates number of log-det estimates with running
	 * averaging of the estimates
	 *
	 * @param num_estimates the number of log-det estimates to be computed
	 * @return the log-det estimates
	 */
	SGVector<float64_t> sample(index_t num_estimates);

	/**
	 * Method that gives num_estimates number of log-det estimates without any
	 * averaging of the estimates
	 *
	 * @param num_estimates the number of log-det estimates to be computed
	 * @return the log-det estimates
	 */
	SGMatrix<float64_t> sample_without_averaging(index_t num_estimates);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LogDetEstimator";
	}

	/** @return trace sampler */
	std::shared_ptr<TraceSampler> get_trace_sampler(void) const;

	/** @return operator function */
	std::shared_ptr<OperatorFunction<float64_t>> get_operator_function(void) const;

private:
	/** the trace sampler */
	std::shared_ptr<TraceSampler> m_trace_sampler;

	/** the linear operator function, which is log in this case */
	std::shared_ptr<OperatorFunction<float64_t>> m_operator_log;

	/** initialize with default values and register params */
	void init();
};

}

#endif // LOG_DET_ESTIMATOR_H_
