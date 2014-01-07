/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _STRINGDISTANCE_H___
#define _STRINGDISTANCE_H___

#include <distance/Distance.h>
#include <features/StringFeatures.h>

namespace shogun
{
/** @brief template class StringDistance */
template <class ST> class CStringDistance : public CDistance
{
	public:
		/** default constructor */
		CStringDistance() : CDistance() {}

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		/* when training data is supplied as both l and r do_init
		 * should be true
		*/
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CDistance::init(l,r);

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
		virtual EFeatureClass get_feature_class() { return C_STRING; }

		/** get feature type the distance can deal with
		 *
		 * @return template-specific feature type
		 */
		virtual EFeatureType get_feature_type();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const {
			return "StringDistance"; }

		/** cleanup distance
		 *
		 * abstract base method
		 */
		virtual void cleanup()=0;

		/** get distance type we are
		 *
		 * abstrace base method
		 *
		 * @return distance type
		 */
		virtual EDistanceType get_distance_type()=0;
};

/** get feature type the DREAL distance can deal with
 *
 * @return feature type DREAL
 */
template<> inline EFeatureType CStringDistance<float64_t>::get_feature_type() { return F_DREAL; }

/** get feature type the ULONG distance can deal with
 *
 * @return feature type ULONG
 */
template<> inline EFeatureType CStringDistance<uint64_t>::get_feature_type() { return F_ULONG; }

/** get feature type the INT distance can deal with
 *
 * @return feature type INT
 */
template<> inline EFeatureType CStringDistance<int32_t>::get_feature_type() { return F_INT; }

/** get feature type the WORD distance can deal with
 *
 * @return feature type WORD
 */
template<> inline EFeatureType CStringDistance<uint16_t>::get_feature_type() { return F_WORD; }

/** get feature type the SHORT distance can deal with
 *
 * @return feature type SHORT
 */
template<> inline EFeatureType CStringDistance<int16_t>::get_feature_type() { return F_SHORT; }

/** get feature type the BYTE distance can deal with
 *
 * @return feature type BYTE
 */
template<> inline EFeatureType CStringDistance<uint8_t>::get_feature_type() { return F_BYTE; }

/** get feature type the CHAR distance can deal with
 *
 * @return feature type CHAR
 */
template<> inline EFeatureType CStringDistance<char>::get_feature_type() { return F_CHAR; }

} // namespace shogun
#endif

