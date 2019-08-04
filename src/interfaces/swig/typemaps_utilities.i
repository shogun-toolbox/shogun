namespace typemap_utils
{
	template <typename T, typename U>
	T& initialize(U& v)
	{
		if constexpr (std::is_pointer_v<U>)
		{
			v = new T{};
			return *v;	
		}
		else
		{
			v = T{};
			return v;
		}
	}

	template <typename T>
	void free_if_pointer(T& t)
	{
		if constexpr (std::is_pointer_v<T>)
			delete t;
	}

	template <typename T, typename U>
	T& cast_deref(U& u)
	{
		if constexpr (std::is_pointer_v<T>)
			return *u;
		else
			return u;
	}
}