/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _MANHATTANWORDDISTANCE_H___
#define _MANHATTANWORDDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/distance/StringDistance.h>

namespace shogun
{
/** @brief class ManhattanWordDistance */
class ManhattanWordDistance: public StringDistance<uint16_t>
{
	public:
		/** default constructor */
		ManhattanWordDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		ManhattanWordDistance(const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r);
		virtual ~ManhattanWordDistance();

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
		 * @return distance type MANHATTANWORD
		 */
		virtual EDistanceType get_distance_type() { return D_MANHATTANWORD; }

		/** get name of the distance
		 *
		 * @return name ManhattanWord
		 */
		virtual const char* get_name() const { return "ManhattanWordDistance"; }

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b);
};
} // namespace shogun
#endif /* _MANHATTANWORDDISTANCE_H___ */
