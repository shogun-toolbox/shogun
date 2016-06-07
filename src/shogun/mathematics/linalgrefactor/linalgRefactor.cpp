#include <shogun/mathematics/linalgrefactor/linalgRefactor.h>

namespace shogun
{

Linalg::Linalg():m_cpubackend(&eigenBackend), m_gpubackend(nullptr){}

Linalg::Linalg(CPUBackend* cpubackend):m_cpubackend(cpubackend), m_gpubackend(nullptr){}

Linalg::Linalg(GPUBackend* gpubackend):m_cpubackend(&eigenBackend), m_gpubackend(gpubackend){}

Linalg::Linalg(CPUBackend* cpubackend, GPUBackend* gpubackend)
:m_cpubackend(cpubackend), m_gpubackend(gpubackend){}

void Linalg::set_cpu_backend(CPUBackend* cpubackend)
{
    m_cpubackend = cpubackend;//const_cast<CPUBackend*>(cpubackend);
}

CPUBackend* Linalg::get_cpu_backend()
{
    return m_cpubackend;
}

void Linalg::set_gpu_backend(GPUBackend* gpubackend)
{
    m_gpubackend = gpubackend;
}

GPUBackend* Linalg::get_gpu_backend()
{
    return m_gpubackend;
}

template <class T>
T Linalg::dot(BaseVector<T> *a, BaseVector<T> *b)
{
    if (a->onGPU() && b->onGPU())
    {
        if (this->hasGPUBackend())
        {
            // do the gpu backend dot product
            // you shouldn't care whether it's viennacl or some other GPU backend.
            return m_gpubackend->dot<T>(static_cast<GPU_Vector<T>&>(*a),
                                            static_cast<GPU_Vector<T>&>(*b));
        } else {
            SG_SERROR("User did not register GPU backend. \n");
            return -1;
        }
    }
    else {
        // take care that the matricies are on the same backend
        if (a->onGPU()){
            SG_SERROR("User did not register GPU backend. \n");
            return -1;
        }
        else if (b->onGPU()) {
            SG_SERROR("User did not register GPU backend. \n");
            return -1;
        }

        // do the non-gpu based default backend:
        // this should be actually as well implemented in a separate class's function and just that being called here:
        // like:
        return m_cpubackend->dot<T>(static_cast<CPUVector<T>&>(*a),
                                        static_cast<CPUVector<T>&>(*b));
    }
}

bool Linalg::hasGPUBackend()
{
    return m_gpubackend != nullptr;
}

template int32_t Linalg::dot<int32_t>(BaseVector<int32_t>* a, BaseVector<int32_t>* b);

}
