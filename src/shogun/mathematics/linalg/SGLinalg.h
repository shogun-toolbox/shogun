#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/LinalgBackendEigen.h>

#ifndef SG_LINALG_H__
#define SG_LINALG_H__

#ifdef HAVE_CXX11

namespace shogun
{

class SGLinalg
{
public:
	SGLinalg()
	{
		cpu_backend = new LinalgBackendEigen();
		gpu_backend = NULL;
	}

	inline void set_cpu_backend(LinalgBackendBase* backend)
	{
		cpu_backend = backend;
	}

	inline LinalgBackendBase* const get_cpu_backend() const
	{
		return cpu_backend;
	}

	inline void set_gpu_backend(LinalgBackendBase* backend)
	{
		gpu_backend = backend;
	}

	inline LinalgBackendBase* const get_gpu_backend() const
	{
		return gpu_backend;
	}

private:
    // cpu is always available (eigen3 or other default/complete implementation)
   LinalgBackendBase* cpu_backend;

   // gpu is NULL until something is registered
   LinalgBackendBase* gpu_backend;
};
}

namespace shogun
{
	extern std::unique_ptr<SGLinalg> sg_linalg;
}
#endif //HAVE_CXX11

#endif //SG_LINALG_H__
