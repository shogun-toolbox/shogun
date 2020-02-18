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
	namespace graph
	{
		enum class element_type
		{
			BOOLEAN = 0,
			INT8 = 1,
			INT16 = 2,
			INT32 = 3,
			INT64 = 4,
			UINT8 = 5,
			UINT16 = 6,
			UINT32 = 7,
			UINT64 = 8,
			FLOAT32 = 9,
			FLOAT64 = 10,
		};

		template <element_type EnumVal>
		struct get_type_from_enum
		{
		};

		template <typename T>
		struct get_enum_from_type
		{
		};

#define TYPE_ENUM_MAPPING(SHOGUN_TYPE, TYPE)                                   \
	template <>                                                                \
	struct get_type_from_enum<SHOGUN_TYPE>                                     \
	{                                                                          \
		using type = TYPE;                                                     \
	};                                                                         \
	template <>                                                                \
	struct get_enum_from_type<TYPE>                                            \
	{                                                                          \
		static constexpr element_type type = SHOGUN_TYPE;                      \
	};

		TYPE_ENUM_MAPPING(element_type::BOOLEAN, bool)
		TYPE_ENUM_MAPPING(element_type::INT8, int8_t)
		TYPE_ENUM_MAPPING(element_type::INT16, int16_t)
		TYPE_ENUM_MAPPING(element_type::INT32, int32_t)
		TYPE_ENUM_MAPPING(element_type::INT64, int64_t)
		TYPE_ENUM_MAPPING(element_type::UINT8, uint8_t)
		TYPE_ENUM_MAPPING(element_type::UINT16, uint16_t)
		TYPE_ENUM_MAPPING(element_type::UINT32, uint32_t)
		TYPE_ENUM_MAPPING(element_type::UINT64, uint64_t)
		TYPE_ENUM_MAPPING(element_type::FLOAT32, float32_t)
		TYPE_ENUM_MAPPING(element_type::FLOAT64, float64_t)

#undef TYPE_ENUM_MAPPING

		inline std::ostream& operator<<(std::ostream& os, element_type type)
		{
			switch (type)
			{
			case element_type::BOOLEAN:
				return os << "bool";
			case element_type::INT8:
				return os << "int8";
			case element_type::INT16:
				return os << "int16";
			case element_type::INT32:
				return os << "int32";
			case element_type::INT64:
				return os << "int64";
			case element_type::UINT8:
				return os << "uint8";
			case element_type::UINT16:
				return os << "uint16";
			case element_type::UINT32:
				return os << "uint32";
			case element_type::UINT64:
				return os << "uint64";
			case element_type::FLOAT32:
				return os << "float32";
			case element_type::FLOAT64:
				return os << "float64";
			}
		}

		inline size_t get_byte_size(element_type type)
		{
			switch (type)
			{
			case element_type::BOOLEAN:
				return sizeof(bool);
			case element_type::INT8:
				return sizeof(int8_t);
			case element_type::INT16:
				return sizeof(int16_t);
			case element_type::INT32:
				return sizeof(int32_t);
			case element_type::INT64:
				return sizeof(int64_t);
			case element_type::UINT8:
				return sizeof(uint8_t);
			case element_type::UINT16:
				return sizeof(uint16_t);
			case element_type::UINT32:
				return sizeof(uint32_t);
			case element_type::UINT64:
				return sizeof(uint64_t);
			case element_type::FLOAT32:
				return sizeof(float32_t);
			case element_type::FLOAT64:
				return sizeof(float64_t);
			}
		}
	} // namespace graph
} // namespace shogun

#endif