#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalgrefactor/GPU_Vector.h>
#include <memory>

#ifdef HAVE_VIENNACL
#include <viennacl/vector.hpp>
#endif

#ifndef GPU_ARRAY_H__
#define GPU_ARRAY_H__

namespace shogun
{

template <class T>
struct GPU_Vector<T>::GPUArray
{

#ifdef HAVE_VIENNACL
    typedef viennacl::backend::mem_handle VCLMemoryArray;
    typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;

    std::shared_ptr<VCLMemoryArray> GPUptr;
    index_t vlen;
    index_t offset;

    GPUArray(const SGVector<T> &vector);
    GPUArray(const GPU_Vector<T>::GPUArray &array);
    VCLVectorBase GPUvec();

//#elif HAVE_CUDA
//    shared_ptr<CUDAArray> GPUptr;
#else
    GPUArray(const SGVector<T> &vector);
    //GPUArray(const GPU_Vector<T>::GPUArray &array);
#endif
};
}

#endif
