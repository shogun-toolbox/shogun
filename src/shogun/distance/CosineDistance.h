/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _COSINEDISTANCE_H___
#define _COSINEDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class CosineDistance
 *
 * The Cosine distance is obtained by using the Cosine similarity (Orchini
 * similarity, angular similarity, normalized dot product), which
 * measures similarity between two vectors by finding their angle.
 * An extension to the Cosine similarity yields the Tanimoto coefficient.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = 1 - \frac{\sum_{i=1}^{n}\bf{x_{i}}\bf{x'_{i}}}
 *  {\sqrt{\sum_{i=1}^{n} x_{i}^2 \sum_{i=1}^{n} {x'}_{i}^2}} \quad x,x' \in R^{n}
 * \f]
 *
 * @see <a href="http://en.wikipedia.org/wiki/Cosine_similarity"> Wikipedia:
 * Cosine similarity </a>
 * @see CTanimotoDistance
 */
class CosineDistance: public DenseDistance<float64_t>
{
	public:
		/** default constructor */
		CosineDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CosineDistance(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r);
		virtual ~CosineDistance();

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
		 * @return distance type COSINE
		 */
		virtual EDistanceType get_distance_type() { return D_COSINE; }

		/** get name of the distance
		 *
		 * @return name Cosine distance
		 */
		virtual const char* get_name() const { return "CosineDistance"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};

} // namespace shogun
#endif /* _COSINEDISTANCE_H___ */
