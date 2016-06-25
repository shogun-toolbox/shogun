#include <shogun/lib/config.h>

#include <shogun/lib/memory.h>
#include <shogun/lib/common.h>

#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/LinalgBackendEigen.h>

#include <memory>

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
		cpu_backend = std::unique_ptr<LinalgBackendBase>(new LinalgBackendEigen());
		gpu_backend = nullptr;
	}

	~SGLinalg()
	{
	}

	void set_cpu_backend(LinalgBackendBase* backend)
	{
		cpu_backend = std::unique_ptr<LinalgBackendBase>(backend);
	}

	LinalgBackendBase* const get_cpu_backend() const
	{
		return cpu_backend.get();
	}

	void set_gpu_backend(LinalgBackendBase* backend)
	{
		gpu_backend = std::unique_ptr<LinalgBackendBase>(backend);
	}

	LinalgBackendBase* const get_gpu_backend() const
	{
		return gpu_backend.get();
	}

private:
	// cpu is always available (eigen3 or other default/complete implementation)
	std::unique_ptr<LinalgBackendBase> cpu_backend;

	// gpu is NULL until something is registered
	std::unique_ptr<LinalgBackendBase> gpu_backend;
};
}

namespace shogun
{
	extern std::unique_ptr<SGLinalg> sg_linalg;
}
#endif //HAVE_CXX11

#endif //SG_LINALG_H__
