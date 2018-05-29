/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <shogun/preprocessor/StringPreprocessor.h>

namespace shogun
{

	template <>
	inline EFeatureType CStringPreprocessor<uint64_t>::get_feature_type()
	{
		return F_ULONG;
	}

	template <>
	inline EFeatureType CStringPreprocessor<int64_t>::get_feature_type()
	{
		return F_LONG;
	}

	template <>
	inline EFeatureType CStringPreprocessor<uint32_t>::get_feature_type()
	{
		return F_UINT;
	}

	template <>
	inline EFeatureType CStringPreprocessor<int32_t>::get_feature_type()
	{
		return F_INT;
	}

	template <>
	inline EFeatureType CStringPreprocessor<uint16_t>::get_feature_type()
	{
		return F_WORD;
	}

	template <>
	inline EFeatureType CStringPreprocessor<int16_t>::get_feature_type()
	{
		return F_WORD;
	}

	template <>
	inline EFeatureType CStringPreprocessor<uint8_t>::get_feature_type()
	{
		return F_BYTE;
	}

	template <>
	inline EFeatureType CStringPreprocessor<int8_t>::get_feature_type()
	{
		return F_BYTE;
	}

	template <>
	inline EFeatureType CStringPreprocessor<char>::get_feature_type()
	{
		return F_CHAR;
	}

	template <>
	inline EFeatureType CStringPreprocessor<bool>::get_feature_type()
	{
		return F_BOOL;
	}

	template <>
	inline EFeatureType CStringPreprocessor<float32_t>::get_feature_type()
	{
		return F_SHORTREAL;
	}

	template <>
	inline EFeatureType CStringPreprocessor<float64_t>::get_feature_type()
	{
		return F_DREAL;
	}

	template <>
	inline EFeatureType CStringPreprocessor<floatmax_t>::get_feature_type()
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

		auto string_list = string_features->get_features();

		apply_to_string_list(string_list);

		SG_UNREF(features);
		return string_features;
	}

	template <class ST>
	bool CStringPreprocessor<ST>::apply_to_string_features(CFeatures* features)
	{
		transform(features);
		return true;
	}

	template class CStringPreprocessor<bool>;
	template class CStringPreprocessor<char>;
	template class CStringPreprocessor<int8_t>;
	template class CStringPreprocessor<uint8_t>;
	template class CStringPreprocessor<int16_t>;
	template class CStringPreprocessor<uint16_t>;
	template class CStringPreprocessor<int32_t>;
	template class CStringPreprocessor<uint32_t>;
	template class CStringPreprocessor<int64_t>;
	template class CStringPreprocessor<uint64_t>;
	template class CStringPreprocessor<float32_t>;
	template class CStringPreprocessor<float64_t>;
	template class CStringPreprocessor<floatmax_t>;
}
