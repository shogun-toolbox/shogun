#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

#include <iostream>

namespace shogun
{

namespace linalgns
{

template <typename Type>
Type dot(const SGVector<Type>& a, const SGVector<Type>& b)
{
    if (a.on_gpu() && b.on_gpu())
	{
		std::cerr << "gpu.dot" << std::endl;
		return sg_linalg->get_gpu_backend()->dot(a, b);
	}
    else if (a.on_gpu() || b.on_gpu())
        SG_SERROR("Vectors must be on the same backend.\n");
	std::cerr << "cpu.dot" << std::endl;
    return sg_linalg->get_cpu_backend()->dot(a, b);
}

LinalgBackendBase* const get_gpu_backend()
{
	return sg_linalg->get_gpu_backend();
}

template <typename T>
SGVector<T> to_gpu(const SGVector<T>& vector)
{
	REQUIRE(!vector.on_gpu(), "Already on GPU");
    LinalgBackendBase* gpu_backend = get_gpu_backend();
    if (gpu_backend)
		return SGVector<T>(gpu_backend->to_gpu(vector), vector.vlen);
    else
    {
        SG_SWARNING("Trying to run GPU code without GPU backend registered.");
        return vector;
    }
}

template <typename T>
SGVector<T> from_gpu(const SGVector<T>& vector)
{
    LinalgBackendBase* gpu_backend = get_gpu_backend();
    if (gpu_backend)
	{
		T* data;
		data = SG_MALLOC(T, vector.vlen);
		gpu_backend->from_gpu(vector, data);
		return SGVector<T>(data, vector.vlen);
	}

    else
    {
        SG_SWARNING("Trying to run GPU code without GPU backend registered.");
        return vector;
    }
}

}

}
