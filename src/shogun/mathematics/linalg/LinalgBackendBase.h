#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/GPUMemoryBase.h>
#include <memory>

#ifndef Linalg_Backend_Base_H__
#define Linalg_Backend_Base_H__

#ifdef HAVE_CXX11

namespace shogun
{

class LinalgBackendBase
{
public:
	// macro to avoid templating
	#define BACKEND_GENERIC_DOT(Type) \
	virtual Type dot(const SGVector<Type>&, const SGVector<Type>&) const \
	{  \
		SG_SNOTIMPLEMENTED; \
	}

	BACKEND_GENERIC_DOT(float32_t);
	BACKEND_GENERIC_DOT(float64_t);
	BACKEND_GENERIC_DOT(int32_t);

	#define BACKEND_GENERIC_TO_GPU(Type) \
	virtual GPUMemoryBase<Type>* to_gpu(const SGVector<Type>&) const \
	{  \
		SG_SWARNING("BASE BACKEND.\n") \
		SG_SNOTIMPLEMENTED; \
	}

	BACKEND_GENERIC_TO_GPU(float32_t);
	BACKEND_GENERIC_TO_GPU(float64_t);
	BACKEND_GENERIC_TO_GPU(int32_t);

	#define BACKEND_GENERIC_FROM_GPU(Type) \
	virtual void from_gpu(const SGVector<Type>&, Type* data) const \
	{  \
		SG_SNOTIMPLEMENTED; \
	}

	BACKEND_GENERIC_FROM_GPU(float32_t);
	BACKEND_GENERIC_FROM_GPU(float64_t);
	BACKEND_GENERIC_FROM_GPU(int32_t);

	virtual const char* get_name() const = 0;
};

}

#endif // HAVE_CXX11

#endif //Linalg_Backend_Base_H__
