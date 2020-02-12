/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTYPES_H_
#define SHOGUNTYPES_H_

#ifdef USE_NGRAPH
#include <ngraph/ngraph.hpp>
#endif

namespace shogun
{
	enum class element_type
	{
		FLOAT32 = 0,
		FLOAT64 = 1
	};

	template <element_type EnumVal>
	struct get_type_from_enum
	{
	};

	template <>
	struct get_type_from_enum<element_type::FLOAT32>
	{
		using type = float32_t;
	};

	template <>
	struct get_type_from_enum<element_type::FLOAT64>
	{
		using type = float64_t;
	};

	template <typename T>
	struct get_enum_from_type
	{
	};

	template <>
	struct get_enum_from_type<float32_t>
	{
		element_type type = element_type::FLOAT32;
	};

	template <>
	struct get_enum_from_type<float64_t>
	{
		element_type type = element_type::FLOAT64;
	};

#ifdef USE_NGRAPH
	element_type get_enum_from_ngraph(ngraph::element::Type_t type)
	{
		switch (type)
		{
		case ngraph::element::Type_t::f32:
			return element_type::FLOAT32;
		case ngraph::element::Type_t::f64:
			return element_type::FLOAT64;
		}
	}
#endif
} // namespace shogun

#endif