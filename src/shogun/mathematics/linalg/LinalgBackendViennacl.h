#include <shogun/mathematics/linalg/LinalgBackendBase.h>

#ifndef Linalg_Backend_ViennaCL_H__
#define Linalg_Backend_ViennaCL_H__

#ifdef HAVE_CXX11
#ifdef HAVE_VIENNACL

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <shogun/mathematics/linalg/GPUMemoryViennaCL.h>

namespace shogun
{

class LinalgBackendViennaCL : public LinalgBackendBase
{
public:
    #define BACKEND_GENERIC_DOT(Type) \
	virtual Type dot(const SGVector<Type>& a, const SGVector<Type>& b) \
    {  \
        return dot_impl(a, b);  \
    }

    BACKEND_GENERIC_DOT(float32_t);
    BACKEND_GENERIC_DOT(float64_t);
    BACKEND_GENERIC_DOT(int32_t);

    #define BACKEND_GENERIC_TO_GPU(Type) \
	virtual GPUMemoryBase<Type>* to_gpu(const SGVector<Type>& vector) \
    {  \
        return to_gpu_impl(vector);  \
    }

    BACKEND_GENERIC_TO_GPU(float32_t);
    BACKEND_GENERIC_TO_GPU(float64_t);
    BACKEND_GENERIC_TO_GPU(int32_t);

	#define BACKEND_GENERIC_FROM_GPU(Type) \
	virtual void from_gpu(const SGVector<Type>& vector, Type* data) \
    {  \
        return from_gpu_impl(vector, data);  \
    }

    BACKEND_GENERIC_FROM_GPU(float32_t);
    BACKEND_GENERIC_FROM_GPU(float64_t);
    BACKEND_GENERIC_FROM_GPU(int32_t);

	/** @return object name */
	virtual const char* get_name() const { return "ViennaCL"; }

private:
    template <typename T>
    T dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
    {

        // we know that the memory is of Vienna type as it was done with this backend
        // if the developer is YOLO and changes the gpu backend after a gpu transfer and calls dot with
        // he DESERVES a crash
        GPUMemoryViennaCL<T>* a_gpu=static_cast<GPUMemoryViennaCL<T>*>(a.gpu_vector);
        GPUMemoryViennaCL<T>* b_gpu=static_cast<GPUMemoryViennaCL<T>*>(b.gpu_vector);

        return viennacl::linalg::inner_prod(a_gpu->data(), b_gpu->data());
    }

    template <typename T>
    GPUMemoryBase<T>* to_gpu_impl(const SGVector<T>& vector) const \
    {
		return (new GPUMemoryViennaCL<T>(vector));
	}


	template <typename T>
	void from_gpu_impl(const SGVector<T>& vector, T* data) const \
	{
		vector.gpu_vector->transfer_to_CPU(data);
	}
};

}
#endif //HAVE_VIENNACL
#endif //HAVE_CXX11

#endif //Linalg_Backend_ViennaCL_H__
