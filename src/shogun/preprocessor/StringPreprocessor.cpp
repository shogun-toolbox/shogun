/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <shogun/preprocessor/StringPreprocessor.h>

namespace shogun
{

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<uint64_t>::get_feature_type()
	{
		return F_ULONG;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<int64_t>::get_feature_type()
	{
		return F_LONG;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<uint32_t>::get_feature_type()
	{
		return F_UINT;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<int32_t>::get_feature_type()
	{
		return F_INT;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<uint16_t>::get_feature_type()
	{
		return F_WORD;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<int16_t>::get_feature_type()
	{
		return F_WORD;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<uint8_t>::get_feature_type()
	{
		return F_BYTE;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<int8_t>::get_feature_type()
	{
		return F_BYTE;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<char>::get_feature_type()
	{
		return F_CHAR;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<bool>::get_feature_type()
	{
		return F_BOOL;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<float32_t>::get_feature_type()
	{
		return F_SHORTREAL;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<float64_t>::get_feature_type()
	{
		return F_DREAL;
	}

	template <>
	SHOGUN_EXPORT EFeatureType CStringPreprocessor<floatmax_t>::get_feature_type()
	{
		return F_LONGREAL;
	}

	template <class ST>
	CFeatures*
	CStringPreprocessor<ST>::transform(CFeatures* features, bool inplace)
	{
		REQUIRE(
		    features->get_feature_class() == C_STRING,
		    "Provided features (%d) "
		    "has to be of C_STRING (%d) class!\n",
		    features->get_feature_class(), C_STRING);

		SG_REF(features);

		// We don't support stealing underlying data for StringFeatures yet.
		// Currently StringPreprocessors modify StringFeatures in place
		// directly.
		auto string_features = features->as<CStringFeatures<ST>>();

		if (!inplace)
		{
			// this actually creates a deep copy
			string_features = new CStringFeatures<ST>(*string_features);
		}

		auto string_list = string_features->get_string_list();

		apply_to_string_list(string_list);

		SG_UNREF(features);
		return string_features;
	}

	template class SHOGUN_EXPORT CStringPreprocessor<bool>;
	template class SHOGUN_EXPORT CStringPreprocessor<char>;
	template class SHOGUN_EXPORT CStringPreprocessor<int8_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<uint8_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<int16_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<uint16_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<int32_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<uint32_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<int64_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<uint64_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<float32_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<float64_t>;
	template class SHOGUN_EXPORT CStringPreprocessor<floatmax_t>;
}
