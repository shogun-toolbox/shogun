/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Evan Shelhamer, Ariane Paola Gomes,
 *          Sergey Lisitsyn
 */

#ifndef _TANIMOTODISTANCE_H___
#define _TANIMOTODISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class Tanimoto coefficient
 *
 * The Tanimoto distance/coefficient (extended Jaccard coefficient)
 * is obtained by extending the cosine similarity.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = \frac{\sum_{i=1}^{n}x_{i}x'_{i}}{
 *  \sum_{i=1}^{n}x_{i}x_{i}x'_{i}x'_{i}-x_{i}x'_{i}}
 *  \quad x,x' \in R^{n}
 * \f]
 *
 * @see <a href="http://en.wikipedia.org/wiki/Jaccard_index">Wikipedia:
 * Tanimoto coefficient</a>
 * @see CCosineDistance
 */
class TanimotoDistance: public DenseDistance<float64_t>
{
	public:
		/** default constructor */
		TanimotoDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		TanimotoDistance(std::shared_ptr<DenseFeatures<float64_t>> l, std::shared_ptr<DenseFeatures<float64_t>> r);
		virtual ~TanimotoDistance();

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** cleanup distance */
		virtual void cleanup();

		/** get distance type we are
		 *
		 * @return distance type TANIMOTO
		 */
		virtual EDistanceType get_distance_type() { return D_TANIMOTO; }

		/** get name of the distance
		 *
		 * @return name Tanimoto coefficient/distance
		 */
		virtual const char* get_name() const { return "TanimotoDistance"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};
} // namespace shogun
#endif /* _TANIMOTODISTANCE_H___ */
