/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _CANBERRAWORDDISTANCE_H___
#define _CANBERRAWORDDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/distance/StringDistance.h>

namespace shogun
{
/**@brief class CanberraWordDistance */
class CCanberraWordDistance: public CStringDistance<uint16_t>
{
	public:
		/** default constructor */
		CCanberraWordDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CCanberraWordDistance(CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r);
		virtual ~CCanberraWordDistance();

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
		 * @return distance type CHEBYSHEW
		 */
		virtual EDistanceType get_distance_type() { return D_CANBERRAWORD; }

		/** get name of the distance
		 *
		 * @return name Chebyshew-Metric
		 */
		virtual const char* get_name() const { return "CanberraWordDistance"; }

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b);
};
} // namespace shogun
#endif /* _CANBERRAWORDDISTANCE_H___ */
