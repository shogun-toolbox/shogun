/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn,
 *          Chiyuan Zhang
 */

#ifndef _MINKOWSKIMETRIC_H___
#define _MINKOWSKIMETRIC_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class MinkowskiMetric
 *
 * The Minkowski metric is one general class of metrics for a
 * \f$\displaystyle R^{n}\f$ feature space also referred as
 * the \f$\displaystyle L_{k} \f$ norm.
 *
 * \f[ \displaystyle
 *  d(\bf{x},\bf{x'}) = (\sum_{i=1}^{n} |\bf{x_{i}}-\bf{x'_{i}}|^{k})^{\frac{1}{k}}
 *  \quad x,x' \in R^{n}
 * \f]
 *
 * special cases:
 * -# \f$\displaystyle L_{1} \f$ norm: Manhattan distance @see CManhattanMetric
 * -# \f$\displaystyle L_{2} \f$ norm: Euclidean distance @see EuclideanDistance
 *
 * Note that the Minkowski distance tends to the Chebyshew distance for
 * increasing \f$k\f$.
 *
 * @see <a href="http://en.wikipedia.org/wiki/Distance">Wikipedia: Distance</a>
 */
class MinkowskiMetric: public DenseDistance<float64_t>
{
	public:
		/** default constructor  */
		MinkowskiMetric();

		/** constructor
		 *
		 * @param k parameter k
		 */
		MinkowskiMetric(float64_t k);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param k parameter k
		 */
		MinkowskiMetric(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r, float64_t k);
		~MinkowskiMetric() override;

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** cleanup distance */
		void cleanup() override;

		/** get distance type we are
		 *
		 * @return distance type MINKOWSKI
		 */
		EDistanceType get_distance_type() override { return D_MINKOWSKI;}

		/** get name of the distance
		 *
		 * @return name Minkowski-Metric
		 */
		const char* get_name() const override { return "MinkowskiMetric"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;

	private:
		void init();

	protected:
		/** parameter k */
		float64_t k;
};

} // namespace shogun
#endif /* _MINKOWSKIMETRIC_H___ */
