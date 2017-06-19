#include <shogun/lib/DataType.h>
#include <shogun/mathematics/linalg/LinalgNamespaceMatrix.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#define DEFINE_SWITCH(PTYPE, FUNC, ...) \
switch (PTYPE) \
{ \
	case PT_FLOAT32: \
		FUNC<float32_t>(__VA_ARGS__); \
		break; \
	case PT_FLOAT64: \
		FUNC<float64_t>(__VA_ARGS__); \
		break; \
	case PT_FLOATMAX: \
		FUNC<floatmax_t>(__VA_ARGS__); \
		break; \
	default: \
		SG_SERROR("Unsupported data type\n"); \
}

namespace shogun
{

namespace linalg
{

template <typename T>
static inline void add_templated(const Matrix& a, const Matrix& b, Matrix& result)
{
	SGMatrix<T> a_sg(a), b_sg(b), result_sg(result);
	add(a_sg, b_sg, result_sg);
}

void add(const Matrix& a, const Matrix& b, Matrix& result)
{
	DEFINE_SWITCH(a.ptype(), add_templated, a, b, result);
}

}

}

#undef DEFINE_SWITCH