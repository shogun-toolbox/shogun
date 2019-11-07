/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Sergey Lisitsyn, Chiyuan Zhang, Heiko Strathmann,
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef STOCHASTICPROXIMITYEMBEDDING_H_
#define STOCHASTICPROXIMITYEMBEDDING_H_
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

/** Stochastic Proximity Embedding (SPE) strategy */
enum ESPEStrategy
{
	SPE_GLOBAL,
	SPE_LOCAL,
};

/** @brief class StochasticProximityEmbedding used to construct embeddings of data using
 * the Stochastic Proximity algorithm.
 *
 * Agrafiotis, D. K. (2002)
 * Stochastic Proximity Embedding
 * Retrieved from:
 * http://www.dimitris-agrafiotis.com/Papers/jcc20078.pdf
 *
 * This class provides two different strategies for the computation of the embedding.
 * In each iteration, both strategies choose two sets of feature vectors whose
 * representation in the embedded space is updated. The first set is randomly chosen
 * in both strategies. However, the second set is obtained differently depending on
 * the strategy used. In the SPE_GLOBAL strategy, the second set is still
 * chosen at random. On the other hand, if SPE_LOCAL is used, first of all, the
 * K-Nearest Neighbors  of each of the feature vectors in the first set is obtained
 * and secondly, a number of feature vectors among these K-Nearest Neighbors is chosen
 * to form the second set.
 *
 * The parameter K for K-Nearest Neighbors in SPE_LOCAL corresponds to the class member
 * "m_k". Each of the two sets used on every iteration is formed by "m_nupdates" feature
 * vectors. Therefore, the number of feature vectors given must be always at least two
 * times the value of "m_nupdates".
 *
 * In order to avoid problems with memory in case a large number of features vectors is
 * to be embedded, the distance matrix is never computed explicitily. This has the
 * drawback that it is likely that the same distances are computed several times during
 * the process.
 *
 * Uses implementation from the Tapkee library.
 *
 * Only EuclideanDistance distance is supported for the moment.
 *
 */

class StochasticProximityEmbedding : public EmbeddingConverter
{

	public:

		/** constructor */
		StochasticProximityEmbedding();

		/** destructor */
		virtual ~StochasticProximityEmbedding();

		/** apply to features
		 *
		 * @param features features to embed
		 * @return embedding features
		 */
		virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);

		/** setter for number of neighbors k in local strategy
		 *
		 * @param k k value
		 */
		void set_k(int32_t k);

		/** getter for number of neighbors k in local strategy
		 *
		 * @return k value
		 */
		int32_t get_k() const;

		/** setter for strategy parameter
		 *
		 * @param strategy type of SPE strategy
		 */
		void set_strategy(ESPEStrategy strategy);

		/** getter for type of SPE strategy
		 *
		 * @return strategy value
		 */
		ESPEStrategy get_strategy() const;

		/** setter for regularization parameter
		 *
		 * @param tolerance regularization value
		 */
		void set_tolerance(float32_t tolerance);

		/** getter for regularization parameter
		 *
		 * @return regularization value
		 */
		int32_t get_tolerance() const;

		/** setter for number of updates per iteration
		 *
		 * @param nupdates number of updates per SPE iteration
		 */
		void set_nupdates(int32_t nupdates);

		/** getter for number of updates per iteration
		 *
		 * @return nupdates value
		 */
		int32_t get_nupdates() const;

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

		/** get name */
		virtual const char* get_name() const;

	private:

		/** default init */
		void init();

		/** apply embedding to Distance
		 * @param distance TODO Euclidean works fine, check with others
		 * @return new features in the embedded space
		 */
		virtual std::shared_ptr<DenseFeatures<float64_t>> embed_distance(std::shared_ptr<Distance> distance);

	private:

		/** SPE strategy */
		ESPEStrategy m_strategy;

		/** number of neighbours in local strategy */
		int32_t m_k;

		/** regularization parameter */
		float32_t m_tolerance;

		/** number of apdates per SPE iteration */
		int32_t m_nupdates;

		/** maximum number of iterations */
		int32_t m_max_iteration;

};

} /* namespace shogun */


#endif /* STOCHASTICPROXIMITYEMBEDDING_H_ */
