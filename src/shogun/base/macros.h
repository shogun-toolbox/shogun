/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Gil Hoben
 */

#ifndef __SG_MACROS_H__
#define __SG_MACROS_H__

#if defined(__GNUC__) || defined(__APPLE__)
#define SG_FORCED_INLINE inline __attribute__((always_inline))
#define SG_FORCED_PACKED __attribute__((__packed__))
#define SG_ATTRIBUTE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
#define SG_FORCED_INLINE __forceinline
#define SG_FORCED_PACKED
#define SG_ATTRIBUTE_UNUSED
#else
#define SG_FORCED_INLINE
#define SG_FORCED_PACKED
#define SG_ATTRIBUTE_UNUSED
#endif

// a quick macro for making sure that an object
// does not have a copy-ctor and operator=
#define SG_DELETE_COPY_AND_ASSIGN(TypeName)                                    \
	TypeName(const TypeName&) = delete;                                        \
	void operator=(const TypeName&) = delete

#ifdef _MSC_VER

#define VA_NARGS(...)                                                          \
	INTERNAL_EXPAND_ARGS_PRIVATE(INTERNAL_ARGS_AUGMENTER(__VA_ARGS__))
#define INTERNAL_ARGS_AUGMENTER(...) unused, __VA_ARGS__
#define INTERNAL_EXPAND(x) x
#define INTERNAL_EXPAND_ARGS_PRIVATE(...)                                      \
	INTERNAL_EXPAND(INTERNAL_GET_ARG_COUNT_PRIVATE(                            \
	    __VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define INTERNAL_GET_ARG_COUNT_PRIVATE(                                        \
    _0_, _1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, count, ...)              \
	count

#else

#define VA_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define VA_NARGS(...) VA_NARGS_IMPL(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#endif

#define VARARG_IMPL2(base, count, ...) base##count(__VA_ARGS__)
#define VARARG_IMPL(base, count, ...) VARARG_IMPL2(base, count, __VA_ARGS__)
#define VARARG(base, ...) VARARG_IMPL(base, VA_NARGS(__VA_ARGS__), __VA_ARGS__)

#define VALUE_TO_STRING_MACRO(s) #s

#define SG_ADD_OPTION(...) VARARG(SG_ADD_OPTION, __VA_ARGS__)

#define SG_OPTIONS(...) __VA_ARGS__

#define SG_ADD_OPTION_BASE(param_name, enum_value)                             \
	{                                                                          \
		static_assert(                                                         \
		    std::is_enum<decltype(enum_value)>::value, "Expected an enum!");   \
		if (has(param_name))                                                   \
		{                                                                      \
			m_string_to_enum_map[param_name]                                   \
			                    [VALUE_TO_STRING_MACRO(enum_value)] =          \
			                        enum_value;                                \
		}                                                                      \
		else                                                                   \
		{                                                                      \
			SG_ERROR(                                                          \
			    "Register parameter %s::%s with SG_ADD before adding options", \
			    get_name(), param_name);                                       \
		}                                                                      \
	}

#define SG_ADD_OPTION1(param_name)                                             \
	{                                                                          \
		static_assert(false, "Need to provide enums to add as options");       \
	}

#define SG_ADD_OPTION2(param_name, enum_value)                                 \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
	}

#ifdef _MSC_VER
#define SG_ADD_OPTION3(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION4(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION5(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION6(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION7(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION8(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION9(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION(param_name, __VA_ARGS__)                                 \
	}
#else
#define SG_ADD_OPTION3(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION2(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION4(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION3(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION5(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION4(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION6(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION5(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION7(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION6(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION8(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION7(param_name, __VA_ARGS__)                                 \
	}

#define SG_ADD_OPTION9(param_name, enum_value, ...)                            \
	{                                                                          \
		SG_ADD_OPTION_BASE(param_name, enum_value)                             \
		SG_ADD_OPTION8(param_name, __VA_ARGS__)                                 \
	}
#endif

// TODO: in C++17 add a if constexpr in the macro to support SG_ADD3 and
//  SG_ADD4. This is needed because param_properties is a enum class
//  so will cause issues during compilation if used in the wrong place.
//  In C++17 do this:
//  if constexpr(std::is_same<
//		         decltype(param_properties), ParameterProperties>::value)
//     {
//	       SG_ADD(param, name, description, param_properties)
//         SG_ADD_OPTION(name, __VA_ARGS__)
//     }
//  else
//     {
//         SG_ADD(param, name, description)
//         SG_ADD_OPTION(name, param_properties, __VA_ARGS__)
//     }
#define SG_ADD_OPTIONS(param, name, description, param_properties, ...)        \
	{                                                                          \
		static_assert(                                                         \
		    std::is_same<                                                      \
		        typename std::remove_pointer<decltype(param)>::type,           \
		        machine_int_t>::value,                                         \
		    "SG_ADD_OPTIONS requires param of type machine_int_t");            \
		static_assert(                                                         \
		    std::is_same<                                                      \
		        decltype(param_properties), ParameterProperties>::value,       \
		    "SG_ADD_OPTIONS requires ParameterProperties in the fourth "       \
		    "argument position");                                              \
		SG_ADD(param, name, description, param_properties);                    \
		SG_ADD_OPTION(name, __VA_ARGS__);                                      \
	}
#endif
