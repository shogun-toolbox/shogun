/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Soeren Sonnenburg,
 *          Evan Shelhamer
 */

#ifndef LAPLACIANEIGENMAPS_H_
#define LAPLACIANEIGENMAPS_H_
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class Features;
class Distance;

/** @brief class LaplacianEigenmaps used to construct embeddings of
 * data using Laplacian Eigenmaps algorithm as described in:
 *
 * Belkin, M., & Niyogi, P. (2002).
 * Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering.
 * Science, 14, 585-591. MIT Press.
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.9400&rep=rep1&type=pdf
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','laplacian_eigenmaps',k,width);
 *
 */
class LaplacianEigenmaps: public EmbeddingConverter
{
public:

	/** constructor */
	LaplacianEigenmaps();

	/** destructor */
	virtual ~LaplacianEigenmaps();

	/** apply to features
	 * @param features to embed
	 * @return embedded features
	 */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);

	/** embed distance
	 * @param distance to use for embedding
	 */
	virtual std::shared_ptr<DenseFeatures<float64_t>> embed_distance(std::shared_ptr<Distance> distance);

	/** setter for K parameter
	 * @param k k value
	 */
	void set_k(int32_t k);

	/** getter for K parameter
	 * @return k value
	 */
	int32_t get_k() const;

	/** setter for TAU parameter
	 * @param tau tau value
	 */
	void set_tau(float64_t tau);

	/** getter for TAU parameter
	 * @return tau value
	 */
	float64_t get_tau() const;

	/** get name */
	virtual const char* get_name() const;

protected:

	/** init */
	void init();

protected:

	/** number of neighbors */
	int32_t m_k;

	/** tau parameter of heat distribution */
	float64_t m_tau;

};
}

#endif /* LAPLACIANEIGENMAPS_H_ */
