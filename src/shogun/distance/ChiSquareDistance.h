/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Sergey Lisitsyn,
 *          Viktor Gal
 */

#ifndef __CHISQUAREDISTANCE_H__
#define __CHISQUAREDISTANCE_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/DenseDistance.h>

namespace shogun
{
/** @brief class ChiSquareDistance
 *
 * This implementation of \f$\chi^{2}\f$ distance extends the
 * concept of \f$\chi^{2}\f$ metric to negative values.
 *
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = \sum_{i=1}^{n}\frac{(x_{i}-x'_{i})^2}
 *  {|x_{i}|+|x'_{i}|} \quad \bf{x},\bf{x'} \in R^{n}
 * \f]
 *
 * @see K. Rieck, P. Laskov. Linear-Time Computation of Similarity Measures
 * for Sequential Data. Journal of Machine Learning Research, 9:23--48,2008.
 */
class ChiSquareDistance: public DenseDistance<float64_t>
{
	public:
		/** default constructor */
		ChiSquareDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		ChiSquareDistance(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r);
		~ChiSquareDistance() override;

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** cleanup distance */
		void cleanup() override;

		/** get distance type we are
		 *
		 * @return distance type CHISQUARE
		 */
		EDistanceType get_distance_type() override { return D_CHISQUARE; }

		/** get name of the distance
		 *
		 * @return name Chi-square distance
		 */
		const char* get_name() const override { return "ChiSquareDistance"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b) override;
};

} // namespace shogun
#endif /* _CHISQUAREDISTANCE_H___ */
