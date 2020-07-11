/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _REALDISTANCE_H__
#define _REALDISTANCE_H__

#include <shogun/lib/config.h>

#include <shogun/distance/DenseDistance.h>
#include <shogun/lib/common.h>

namespace shogun
{
/** @brief class RealDistance */
class RealDistance : public DenseDistance<float64_t>
{
public:
	/** default constructor */
	RealDistance() : DenseDistance<float64_t>() {}

	/** init distance
	 *
	 * @param l features of left-hand side
	 * @param r features of right-hand side
	 * @return if init was successful
	 */
	bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override
	{
		DenseDistance<float64_t>::init(l,r);

		ASSERT(l->get_feature_type()==F_DREAL)
		ASSERT(r->get_feature_type()==F_DREAL)

		return true;
	}

	/** get feature type the distance can deal with
	 *
	 * @return feature type DREAL
	 */
	EFeatureType get_feature_type() override { return F_DREAL; }

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 * @return name of the SGSerializable
	 */
	const char* get_name() const override { return "RealDistance"; }

	/** cleanup distance
	 *
	 * abstract base method
	 */
	void cleanup() override =0;

	/** get distance type we are
	 *
	 * abstrace base method
	 *
	 * @return distance type
	 */
	EDistanceType get_distance_type() override =0 ;

protected:
	/// compute distance function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	float64_t compute(int32_t x, int32_t y) override =0;
};
} // namespace shogun
#endif
