#include <shogun/mathematics/linalgrefactor/SGLinalg.h>

namespace shogun
{

SGLinalg::SGLinalg():m_cpubackend(std::unique_ptr<CPUBackend>(new CPUBackend)), m_gpubackend(nullptr)
{
}

SGLinalg::SGLinalg(std::unique_ptr<CPUBackend> cpubackend)
:m_cpubackend(std::unique_ptr<CPUBackend>(new CPUBackend(*(cpubackend)))), m_gpubackend(nullptr)
{
}

SGLinalg::SGLinalg(std::unique_ptr<GPUBackend> gpubackend)
:m_cpubackend(std::unique_ptr<CPUBackend>(new CPUBackend)),
 m_gpubackend(std::unique_ptr<GPUBackend>(new GPUBackend(*(gpubackend))))
 {
 }

SGLinalg::SGLinalg(std::unique_ptr<CPUBackend> cpubackend, std::unique_ptr<GPUBackend> gpubackend)
:m_cpubackend(std::unique_ptr<CPUBackend>(new CPUBackend(*(cpubackend)))),
 m_gpubackend(std::unique_ptr<GPUBackend>(new GPUBackend(*(gpubackend))))
 {
 }

void SGLinalg::set_cpu_backend(std::unique_ptr<CPUBackend> cpubackend)
{
	m_cpubackend.reset(new CPUBackend(*(cpubackend)));
}

CPUBackend* SGLinalg::get_cpu_backend()
{
	return m_cpubackend.get();
}

void SGLinalg::set_gpu_backend(std::unique_ptr<GPUBackend> gpubackend)
{
	m_gpubackend.reset(new GPUBackend(*(gpubackend)));
}

GPUBackend* SGLinalg::get_gpu_backend()
{
	return m_gpubackend.get();
}

template <class T>
T SGLinalg::dot(BaseVector<T> *a, BaseVector<T> *b) const
{
	if (a->onGPU() && b->onGPU())
	{
		if (this->hasGPUBackend())
		{
			// do the gpu backend dot product
			// you shouldn't care whether it's viennacl or some other GPU backend.
			return m_gpubackend->dot<T>(static_cast<GPUVector<T>&>(*a),
                        	                    static_cast<GPUVector<T>&>(*b));
		} 
		else 
		{
			SG_SERROR("User did not register GPU backend. \n");
 			return -1;
		}
	}
	else 
	{
		// take care that the matricies are on the same backend
		if (a->onGPU())
		{
			SG_SERROR("User did not register GPU backend. \n");
			return -1;
		}
		else if (b->onGPU()) 
		{
			SG_SERROR("User did not register GPU backend. \n");
			return -1;
		}
		return m_cpubackend->dot<T>(static_cast<CPUVector<T>&>(*a),
                                        static_cast<CPUVector<T>&>(*b));
	}
}

bool SGLinalg::hasGPUBackend() const
{
	return m_gpubackend != nullptr;
}

template int32_t SGLinalg::dot<int32_t>(BaseVector<int32_t> *a, BaseVector<int32_t> *b) const;
template float32_t SGLinalg::dot<float32_t>(BaseVector<float32_t> *a, BaseVector<float32_t> *b) const;
}
