#include <shogun/mathematics/linalgrefactor/linalgRefactor.h>

namespace shogun
{

Linalg::Linalg():cpubackend(nullptr), gpubackend(nullptr){}

Linalg::Linalg(CPUBackend* cpu_backend):cpubackend(cpu_backend), gpubackend(nullptr){}

Linalg::Linalg(GPUBackend* gpu_backend):cpubackend(nullptr), gpubackend(gpu_backend){}

Linalg::Linalg(CPUBackend* cpu_backend, GPUBackend* gpu_backend)
:cpubackend(cpu_backend), gpubackend(gpu_backend){}

void Linalg::set_cpu_backend(const CPUBackend* cpu_backend)
{
    this->cpubackend = const_cast<CPUBackend*>(cpu_backend);
}

CPUBackend* Linalg::get_cpu_backend()
{
    return cpubackend;
}

void Linalg::set_gpu_backend(const GPUBackend* gpu_backend)
{
    this->gpubackend = const_cast<GPUBackend*>(gpu_backend);
}

GPUBackend* Linalg::get_gpu_backend()
{
    return gpubackend;
}

template <class T>
T Linalg::dot(BaseVector<T>* a, BaseVector<T>* b)
{
    if (a->onGPU() && b->onGPU())
    {
        if (this->hasGPUBackend())
        {
            // do the gpu backend dot product
            // you shouldn't care whether it's viennacl or some other GPU backend.
            return this->gpubackend->dot<T>(static_cast<GPU_Vector<T>&>(*a),
                                            static_cast<GPU_Vector<T>&>(*b));
        } else {
            SG_SERROR("User did not register GPU backend. \n");
        }
    }
    else {
        // take care that the matricies are on the same backend
        if (a->onGPU()){
            SG_SERROR("User did not register GPU backend. \n");
        }
        else if (b->onGPU()) {
            SG_SERROR("User did not register GPU backend. \n");
        }

        // do the non-gpu based default backend:
        // this should be actually as well implemented in a separate class's function and just that being called here:
        // like:
        return this->cpubackend->dot<T>(static_cast<CPUVector<T>&>(*a),
                                        static_cast<CPUVector<T>&>(*b));
    }
}

bool Linalg::hasGPUBackend()
{
    return gpubackend != nullptr;
}

template int32_t Linalg::dot<int32_t>(BaseVector<int32_t>* a, BaseVector<int32_t>* b);

}
