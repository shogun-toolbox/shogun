/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Bjoern Esser, Kunal Arora, Heiko Strathmann, Fernando Iglesias, 
 *          Sergey Lisitsyn, Soeren Sonnenburg
 */

#ifndef FACTOR_ANALYSIS_H_
#define FACTOR_ANALYSIS_H_
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>

namespace shogun
{

/** @brief The Factor Analysis class is used to embed
 * data using Factor Analysis algorithm.
 *
 * Factor analysis aims at describing how several observed variables are
 * correlated to each other by means of identifying a set of unobserved
 * variables, the so-called factors. Desirably, the number of factors is
 * shorter than the number of observed variables.
 *
 * Factor analysis is an iterative algorithm. First of all the
 * projection matrix is initialized randomly and the factors variance is
 * set to the identity. Then, every iteration consists of the following
 * steps:
 * - Compute the regularized inverse covariance matrix of the projection.
 * - Update the factors variance matrix.
 * - Update the projection matrix.
 * - Check for convergence using the log-likelihood of the model. If the
 *   difference between the current log-likelihood and the previous
 *   iteration's log-likelihood is below a threshold, then the algorithm
 *   has converged.
 *
 * Uses implementation from the Tapkee library.
 *
 * cf. http://tapkee.lisitsyn.me/ <br>
 * cf. http://en.wikipedia.org/wiki/Factor_analysis <br>
 * cf. Spearman, C. (1904). General Intelligence, Objectively Determined
 *     and Measured.
 *     (http://www.mendeley.com/catalog/general-intelligence-objectively-determined-measured/)
 *
 */
class FactorAnalysis : public EmbeddingConverter
{
public:

	/** constructor */
	FactorAnalysis();

	/** destructor */
	virtual ~FactorAnalysis();

	/** get name */
	virtual const char* get_name() const;

	/** apply preprocessor to features
	 *
	 * @param features features to embed
	 */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);

	/** setter for the maximum number of iterations
	 *
	 * @param max_iteration the maximum number of iterations
	 */
	void set_max_iteration(const int32_t max_iteration);

	/** getter for the maximum number of iterations
	 *
	 * @return the maximum number of iterations
	 */
	int32_t get_max_iteration() const;

	/** setter for epsilon, parameter used to check for convergence
	 *
	 * @param epsilon convergence parameter
	 */
	void set_epsilon(const float64_t epsilon);

	/** getter for epsilon, parameter used to check for convergence
	 *
	 * @return value of the convergence parameter
	 */
	float64_t get_epsilon() const;

private:

	/** default init */
	void init();

private:

	/** maximum number of iterations */
	int32_t m_max_iteration;

	/** convergence parameter */
	float64_t m_epsilon;

}; /* class FactorAnalysis */

} /* namespace shogun */

#endif /* FACTOR_ANALYSIS_H_ */
