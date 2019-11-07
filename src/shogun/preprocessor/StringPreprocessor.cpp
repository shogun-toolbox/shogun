/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <shogun/preprocessor/StringPreprocessor.h>

namespace shogun
{

	template <>
	inline EFeatureType StringPreprocessor<uint64_t>::get_feature_type()
	{
		return F_ULONG;
	}

	template <>
	inline EFeatureType StringPreprocessor<int64_t>::get_feature_type()
	{
		return F_LONG;
	}

	template <>
	inline EFeatureType StringPreprocessor<uint32_t>::get_feature_type()
	{
		return F_UINT;
	}

	template <>
	inline EFeatureType StringPreprocessor<int32_t>::get_feature_type()
	{
		return F_INT;
	}

	template <>
	inline EFeatureType StringPreprocessor<uint16_t>::get_feature_type()
	{
		return F_WORD;
	}

	template <>
	inline EFeatureType StringPreprocessor<int16_t>::get_feature_type()
	{
		return F_WORD;
	}

	template <>
	inline EFeatureType StringPreprocessor<uint8_t>::get_feature_type()
	{
		return F_BYTE;
	}

	template <>
	inline EFeatureType StringPreprocessor<int8_t>::get_feature_type()
	{
		return F_BYTE;
	}

	template <>
	inline EFeatureType StringPreprocessor<char>::get_feature_type()
	{
		return F_CHAR;
	}

	template <>
	inline EFeatureType StringPreprocessor<bool>::get_feature_type()
	{
		return F_BOOL;
	}

	template <>
	inline EFeatureType StringPreprocessor<float32_t>::get_feature_type()
	{
		return F_SHORTREAL;
	}

	template <>
	inline EFeatureType StringPreprocessor<float64_t>::get_feature_type()
	{
		return F_DREAL;
	}

	template <>
	inline EFeatureType StringPreprocessor<floatmax_t>::get_feature_type()
	{
		return F_LONGREAL;
	}

	template <class ST>
	std::shared_ptr<Features>
	StringPreprocessor<ST>::transform(std::shared_ptr<Features> features, bool inplace)
	{
		require(
		    features->get_feature_class() == C_STRING,
		    "Provided features ({}) "
		    "has to be of C_STRING ({}) class!",
		    features->get_feature_class(), C_STRING);



		// We don't support stealing underlying data for StringFeatures yet.
		// Currently StringPreprocessors modify StringFeatures in place
		// directly.
		auto string_features = features->as<StringFeatures<ST>>();

		if (!inplace)
		{
			// this actually creates a deep copy
			string_features = std::make_shared<StringFeatures<ST>>(*string_features);
		}

		auto& string_list = string_features->get_string_list();

		apply_to_string_list(string_list);


		return string_features;
	}

	template class StringPreprocessor<bool>;
	template class StringPreprocessor<char>;
	template class StringPreprocessor<int8_t>;
	template class StringPreprocessor<uint8_t>;
	template class StringPreprocessor<int16_t>;
	template class StringPreprocessor<uint16_t>;
	template class StringPreprocessor<int32_t>;
	template class StringPreprocessor<uint32_t>;
	template class StringPreprocessor<int64_t>;
	template class StringPreprocessor<uint64_t>;
	template class StringPreprocessor<float32_t>;
	template class StringPreprocessor<float64_t>;
	template class StringPreprocessor<floatmax_t>;
}
