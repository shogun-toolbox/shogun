/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _STRINGDISTANCE_H___
#define _STRINGDISTANCE_H___

#include <shogun/lib/config.h>

#include <shogun/distance/Distance.h>
#include <shogun/features/StringFeatures.h>

#define IGNORE_IN_CLASSLIST

namespace shogun
{
/** @brief template class StringDistance */
IGNORE_IN_CLASSLIST template <class ST> class StringDistance : public Distance
{
	public:
		/** default constructor */
		StringDistance() : Distance() {}

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		/* when training data is supplied as both l and r do_init
		 * should be true
		*/
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override
		{
			Distance::init(l,r);

			ASSERT(l->get_feature_class()==C_STRING)
			ASSERT(r->get_feature_class()==C_STRING)
			ASSERT(l->get_feature_type()==this->get_feature_type())
			ASSERT(r->get_feature_type()==this->get_feature_type())
			return true;
		}

		/** get feature class the distance can deal with
		 *
		 * @return feature class STRING
		 */
		EFeatureClass get_feature_class() override { return C_STRING; }

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
			return "StringDistance"; }

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
		EDistanceType get_distance_type() override =0;
};

/** get feature type the DREAL distance can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType StringDistance<float64_t>::get_feature_type() { return F_DREAL; }

/** get feature type the ULONG distance can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType StringDistance<uint64_t>::get_feature_type() { return F_ULONG; }

/** get feature type the INT distance can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType StringDistance<int32_t>::get_feature_type() { return F_INT; }

/** get feature type the WORD distance can deal with
 *
 * @return feature type WORD
 */
template<> inline EFeatureType StringDistance<uint16_t>::get_feature_type() { return F_WORD; }

/** get feature type the SHORT distance can deal with
 *
 * @return feature type SHORT
 */
template<> inline EFeatureType StringDistance<int16_t>::get_feature_type() { return F_SHORT; }

/** get feature type the BYTE distance can deal with
 *
 * @return feature type BYTE
 */
template<> inline EFeatureType StringDistance<uint8_t>::get_feature_type() { return F_BYTE; }

/** get feature type the CHAR distance can deal with
 *
 * @return feature type CHAR
 */
template<> inline EFeatureType StringDistance<char>::get_feature_type() { return F_CHAR; }

} // namespace shogun
#endif

