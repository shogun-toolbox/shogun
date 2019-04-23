/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Vladislav Horbatiuk, Soeren Sonnenburg, 
 *          Bjoern Esser
 */

#ifndef MANIFOLDSCULPTING_H_
#define MANIFOLDSCULPTING_H_
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>

namespace shogun
{

/** @brief class ManifoldSculpting used to embed
 * data using manifold sculpting embedding algorithm.
 *
 * Uses implementation from the Tapkee library.
 *
 */
class ManifoldSculpting : public EmbeddingConverter
{
public:

	/** constructor */
	ManifoldSculpting();

	/** destructor */
	virtual ~ManifoldSculpting();

	/** get name */
	virtual const char* get_name() const;

	/** apply preprocessor to features
	 *
	 * @param features features to embed
	 */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);

	/** setter for the k
	 *
	 * @param k the number of neighbors
	 */
	void set_k(const int32_t k);

	/** getter for the number of neighbors
	 *
	 * @return the number of neighbors k
	 */
	int32_t get_k() const;

	/** setter for squishing_rate
	 *
	 * @param squishing_rate the squishing rate
	 */
	void set_squishing_rate(const float64_t squishing_rate);

	/** getter for squishing_rate
	 *
	 * @return squishing_rate
	 */
	float64_t get_squishing_rate() const;

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

private:

	/** default init */
	void init();

private:

	/** k - number of neighbors */
	float64_t m_k;

	/** squishing_rate */
	float64_t m_squishing_rate;

	/** max_iteration - the maximum number of algorithm's
	 * iterations
	 */
	float64_t m_max_iteration;

}; /* class ManifoldSculpting */

} /* namespace shogun */

#endif /* MANIFOLDSCULPTING_H_ */
