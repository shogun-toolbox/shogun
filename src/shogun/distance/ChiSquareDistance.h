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
class SHOGUN_EXPORT CChiSquareDistance: public CDenseDistance<float64_t>
{
	public:
		/** default constructor */
		CChiSquareDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CChiSquareDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r);
		virtual ~CChiSquareDistance();

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** cleanup distance */
		virtual void cleanup();

		/** get distance type we are
		 *
		 * @return distance type CHISQUARE
		 */
		virtual EDistanceType get_distance_type() { return D_CHISQUARE; }

		/** get name of the distance
		 *
		 * @return name Chi-square distance
		 */
		virtual const char* get_name() const { return "ChiSquareDistance"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);
};

} // namespace shogun
#endif /* _CHISQUAREDISTANCE_H___ */
