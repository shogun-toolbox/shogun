#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <memory>
#include <shogun/mathematics/eigen3.h>
#include <iostream>

using namespace shogun;

template <class T>
struct BaseVector
{
    BaseVector(){}
    BaseVector(const SGVector<T> vector){}

    virtual bool onGPU()
    {
        return false;
    }
};

template <class T>
struct CPU_Vector : public BaseVector<T>
{
    //unique_pointer<SGVector<T>> CPUptr;
    SGVector<T> vec;

    CPU_Vector(const SGVector<T> vector)
    {
        //CPUptr = unique_pointer<SGVector<T>>(new SGVector<T>(vec));
        vec = vector;
    }

    bool onGPU()
    {
        return false;
    }
};

template <typename T>
struct GPU_Vector : public BaseVector<T>
{
#ifdef HAVE_VIENNACL
    // unique_pointer<VCLMemoryArray> GPUptr;
    // other gpu related stuff
#endif
    bool onGPU()
    {
        return true;
    }
};

//template <typename T>
class CPUBackend
{
public:
    template <typename T>
    T dot(CPU_Vector<T> a, CPU_Vector<T> b)
    {
        std::cerr << "CPU calculation" << std::endl;
        std::cerr << a.vec.vlen << std::endl;
        std::cerr << b.vec.vlen << std::endl;
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
        Eigen::Map<VectorXt> vec_a = a.vec;
        Eigen::Map<VectorXt> vec_b = b.vec;
        return vec_a.dot(vec_b);
    }

   // similarly, other methods
};

//template <typename T>
class GPUBackend
{
public:
 #ifdef HAVE_VIENNACL
    template <typename T>
    T dot(GPU_Vector<T> a, GPU_Vector<T> b)
    {
        // Dereference a.GPUptr and b.GPUptr to vcl_vector?
        // viennacl::linalg::inner_prod(vcl_vector_a, vcl_vector_b);
        // Transfer back to CPU end???
    }

   // similarly, other methods
 #endif
 };

class LinalgRefactor
{
    std::shared_ptr<CPUBackend> cpubackend;
    std::shared_ptr<GPUBackend> gpubackend;

public:
    LinalgRefactor():cpubackend({}), gpubackend({}){}

    LinalgRefactor(std::shared_ptr<CPUBackend> cpu_backend):cpubackend(cpu_backend), gpubackend({}){}

    LinalgRefactor(std::shared_ptr<GPUBackend> gpu_backend):cpubackend({}), gpubackend(gpu_backend){}

    LinalgRefactor(std::shared_ptr<CPUBackend> cpu_backend, std::shared_ptr<GPUBackend> gpu_backend)
    :cpubackend(cpu_backend), gpubackend(gpu_backend){}

    template <class T>
    T dot(std::shared_ptr<BaseVector<T> > a, std::shared_ptr<BaseVector<T> > b)
    {
        if (a->onGPU() && b->onGPU())
        {
            if (this->hasGPUBackend())
            {
                // do the gpu backend dot product
                // you shouldn't care whether it's viennacl or some other GPU backend.
                return this->gpubackend->dot<T>(*static_cast<GPU_Vector<T>*>(a.get()),
                                                *static_cast<GPU_Vector<T>*>(b.get()));
            } else {
                // either throw a RuntimeException or transfer back the data to cpu
                // throw new RuntimeException("user did not register GPU backend");
            }
        }
        else {
            // take care that the matricies are on the same backend
            if (a->onGPU()){ }//Transfer back to CPU || throw error ??? }
            else if (b->onGPU()) { }//Transfer back to CPU || throw error }

            // do the non-gpu based default backend:
            // this should be actually as well implemented in a separate class's function and just that being called here:
            // like:
            std::cerr << "CPUBackend!" << std::endl;
            //std::cerr << (a->vec).vlen << std::endl;
            //std::cerr << (b->vec).vlen << std::endl;
            return this->cpubackend->dot<T>(*static_cast<CPU_Vector<T>*>(a.get()),
                                            *static_cast<CPU_Vector<T>*>(b.get()));
        }
    }

    bool hasGPUBackend()
    {
        return gpubackend != nullptr;
    }
};
