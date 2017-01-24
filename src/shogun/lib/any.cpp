#include <shogun/lib/any.h>

template<>
Any::EnumType Any::type2enum<int>()
{
	return TYPE_INT;
}

template<>
Any::EnumType Any::type2enum<double>()
{
	constexpr EnumType type = TYPE_DOUBLE;
	return type;
}

template<>
Any::EnumType Any::type2enum<SGVector<int>>()
{
	constexpr EnumType type = TYPE_SGVECTOR_INT;
	return type;
}

template<>
Any::EnumType Any::type2enum<SGVector<double>>()
{
	constexpr EnumType type = TYPE_SGVECTOR_DOUBLE;
	return type;
}
