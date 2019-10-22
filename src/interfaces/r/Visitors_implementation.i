/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

%include "GetterVisitorInterface.i"
%{
	namespace shogun
	{

		#define SG_TO_R_TYPE_STRUCT(SG_TYPE, R_TYPE_VALUE)         \
		template <>                                   			   \
		struct sg_to_r_type<SG_TYPE>                     		   \
		{                                              			   \
			const static SEXPTYPE type = R_TYPE_VALUE;     	       \
		};

		template <typename T>
		struct sg_to_r_type {};
		SG_TO_R_TYPE_STRUCT(bool,          LGLSXP)
		SG_TO_R_TYPE_STRUCT(char,          CHARSXP)
		SG_TO_R_TYPE_STRUCT(int8_t,        INTSXP)
		SG_TO_R_TYPE_STRUCT(uint8_t,       INTSXP)
		SG_TO_R_TYPE_STRUCT(int16_t,       INTSXP)
		SG_TO_R_TYPE_STRUCT(uint16_t,      INTSXP)
		SG_TO_R_TYPE_STRUCT(int32_t,       INTSXP)
		SG_TO_R_TYPE_STRUCT(uint32_t,      INTSXP)
		SG_TO_R_TYPE_STRUCT(int64_t,       INTSXP)
		SG_TO_R_TYPE_STRUCT(uint64_t,      INTSXP)
		SG_TO_R_TYPE_STRUCT(float32_t,     REALSXP)
		SG_TO_R_TYPE_STRUCT(float64_t,     REALSXP)
		SG_TO_R_TYPE_STRUCT(complex128_t,  REALSXP)
		SG_TO_R_TYPE_STRUCT(floatmax_t,    REALSXP)
		// SG_TO_R_TYPE_STRUCT(SGObject*,    EXTPTRSXP) // not sure what to put here
		#undef SG_TO_R_TYPE_STRUCT

		class RVisitor: public GetterVisitorInterface<RVisitor, SEXP>
		{
			friend class GetterVisitorInterface<RVisitor, SEXP>;

		public:
			RVisitor(SEXP& obj): GetterVisitorInterface(obj) {}

		protected:
			template <typename T>
			SEXP create_array(const T* v, const std::vector<std::ptrdiff_t>& dims)
			{
				SEXP result;
				size_t len;
				if constexpr(std::is_same_v<T, SGObject*>)
				{
					error("Cannot handle SGObject arrays!");
					return nullptr;
				}
				else
				{
					if (dims.size() == 1)
					{
						len = dims[0];
						PROTECT(result = Rf_allocVector(sg_to_r_type<T>::type, dims[0]));
					}
					else if (dims.size() == 2)
					{
						len = dims[0] * dims[1];
						PROTECT(result = Rf_allocMatrix(sg_to_r_type<T>::type, dims[0], dims[1]));
					}
					else
						error("Unexpected number of dimensions: {}.", dims.size());

					if constexpr(std::is_same_v<T, bool>)
						std::copy_n(v, len, LOGICAL_POINTER(result));
					else if constexpr(std::is_same_v<T, complex128_t>)
						std::copy_n(reinterpret_cast<Rcomplex*>(const_cast<T*>(v)), len, COMPLEX_POINTER(result));
					else if constexpr(std::is_floating_point_v<T>)
						std::copy_n(v, len, NUMERIC_POINTER(result));
					else
						std::copy_n(v, len, INTEGER_POINTER(result));

					UNPROTECT(1);
					return result;
				}
			}

			template <typename T>
			SEXP create_new_list(size_t size)
			{
				// here we need to know the type
				// we use a vector instead of a list in R ¯\_(ツ)_/¯
				SEXP result;
				if constexpr(std::is_same_v<T, SGObject*>)
				{
					error("Cannot handle SGObject lists!");
					return nullptr;
				}
				else
				{
					// if the element of the list are vectors we need to use VECSXP (STRSXP for SGVector<char>)
					if (m_nested_type == SG_TYPE_TO_INTERFACE::VECTOR || m_nested_type == SG_TYPE_TO_INTERFACE::MATRIX)
					{
						if constexpr(std::is_same_v<T, char>)
							PROTECT(result = Rf_allocVector(STRSXP, size));
						else
							PROTECT(result = Rf_allocVector(VECSXP, size));
					}
					// otherwise we are just creating a list of scalar elements
					else
						PROTECT(result = Rf_allocVector(sg_to_r_type<T>::type, size));
					UNPROTECT(1);
					return result;
				}
			}

			template <typename T>
			void append_to_list(SEXP array, SEXP v, size_t i)
			{
				if constexpr(std::is_same_v<T, SGObject*>)
					error("Cannot handle SGObject lists!");
				// special situation where we set array element to a char pointer (CHARSXP)
				else if constexpr (std::is_same_v<T, char>)
					SET_STRING_ELT(array, i, STRING_ELT(v, 0));
				else
					SET_VECTOR_ELT(array, i, v);
			}

			template <typename T>
			SEXP sg_to_interface(const T* v)
			{
				// table of conversions from C++ to R
				if constexpr(std::is_same_v<T, bool>)
					return Rf_ScalarLogical(*v);
				if constexpr(std::is_same_v<T, int8_t>)
					return Rf_ScalarInteger(static_cast<int>(*v));
				if constexpr(std::is_same_v<T, int16_t>)
					return Rf_ScalarInteger(static_cast<int>(*v));
				if constexpr(std::is_same_v<T, int32_t>)
					return Rf_ScalarInteger(*v);
				if constexpr(std::is_same_v<T, int64_t>)
					return Rf_ScalarInteger(static_cast<int>(*v));
				if constexpr(std::is_same_v<T, float32_t>)
					return Rf_ScalarReal(static_cast<double>(*v));
				if constexpr(std::is_same_v<T, float64_t>)
					return Rf_ScalarReal(static_cast<double>(*v));
				if constexpr(std::is_same_v<T, floatmax_t>)
					return Rf_ScalarReal(static_cast<double>(*v));
				if constexpr(std::is_same_v<T, char>)
					return SWIG_FromCharPtrAndSize(v, strlen(v));
				if constexpr(std::is_same_v<T, uint8_t>)
					return Rf_ScalarInteger(static_cast<int>(*v));
				if constexpr(std::is_same_v<T, uint16_t>)
					return  Rf_ScalarInteger(static_cast<int>(*v));
				if constexpr(std::is_same_v<T, uint32_t>)
					return Rf_ScalarInteger(static_cast<int>(*v));
				if constexpr(std::is_same_v<T, uint64_t>)
					return Rf_ScalarInteger(static_cast<int>(*v));
				if constexpr(std::is_same_v<T, complex128_t>)
					return Rf_ScalarComplex(Rcomplex{v->real(), v->imag()});
				if constexpr(std::is_same_v<T, SGObject*>)
					return SWIG_R_NewPointerObj(SWIG_as_voidptr(*v), SWIGTYPE_p_shogun__CSGObject, 0);
				if constexpr(std::is_same_v<T, std::shared_ptr<SGObject>>)
					return SWIG_R_NewPointerObj(SWIG_as_voidptr(v), SWIGTYPE_p_std__shared_ptrT_shogun__SGObject_t, SWIG_POINTER_OWN);
				error("Cannot handle casting from shogun type {} to R type!", demangled_type<T>().c_str());
			}
		};
	}
%}
