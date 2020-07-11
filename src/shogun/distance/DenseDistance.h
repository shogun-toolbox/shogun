/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _DENSEDISTANCE_H___
#define _DENSEDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/distance/Distance.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
/** @brief template class DenseDistance */
IGNORE_IN_CLASSLIST template <class ST> class DenseDistance : public Distance
{
	public:
		/** default constructor */
		DenseDistance() : Distance() {}

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** get feature class the distance can deal with
		 *
		 * @return feature class DENSE
		 */
		EFeatureClass get_feature_class() override { return C_DENSE; }

		/** get feature type the distance can deal with
		 *
		 * @return template-specific feature type
		 */
		EFeatureType get_feature_type() override;

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		const char* get_name() const override {
			return "DenseDistance"; }

		/** get distance type we are
		 *
		 * abstrace base method
		 *
		 * @return distance type
		 */
		EDistanceType get_distance_type() override =0;
};
} // namespace shogun
#endif
