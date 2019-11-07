/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _HAMMINGWORDDISTANCE_H___
#define _HAMMINGWORDDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/distance/StringDistance.h>

namespace shogun
{
	template <class T> class StringFeatures;

/** @brief class HammingWordDistance */
class HammingWordDistance: public StringDistance<uint16_t>
{
	public:
		/** default constructor  */
		HammingWordDistance();

		/** constructor
		 *
		 * @param use_sign if sign shall be used
		 */
		HammingWordDistance(bool use_sign);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 */
		HammingWordDistance(const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r, bool use_sign);
		virtual ~HammingWordDistance();

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
		 * @return distance type HAMMINGWORD
		 */
		virtual EDistanceType get_distance_type() { return D_HAMMINGWORD; }

		/** get name of the distance
		 *
		 * @return name HammingWord
		 */
		virtual const char* get_name() const { return "HammingWordDistance"; }

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		void init();

	protected:
		/** if sign shall be used */
		bool use_sign;
};
} // namespace shogun
#endif /* _HAMMINGWORDDISTANCE_H___ */
