#include <shogun/mathematics/linalgrefactor/GPU_Vector.h>
#include <shogun/mathematics/linalgrefactor/GPUArray.h>

namespace shogun
{

template <class T>
GPU_Vector<T>::GPU_Vector()
{
    init();
}

template <class T>
GPU_Vector<T>::GPU_Vector(const SGVector<T> &vector)
{
    init();
    vlen = vector.vlen;

#ifdef HAVE_VIENNACL
    gpuarray = std::unique_ptr<GPUArray>(new GPUArray(vector));
#else
    SG_SERROR("User did not register GPU backend. \n");
#endif
}

template <class T>
GPU_Vector<T>::GPU_Vector(const GPU_Vector<T> &vector)
{
    init();
    vlen = vector.vlen;
    offset = vector.offset;
#ifdef HAVE_VIENNACL
    gpuarray = std::unique_ptr<GPUArray>(new GPUArray(*(vector.gpuarray)));
#else
    SG_SERROR("User did not register GPU backend. \n");
#endif
}

template <class T>
void GPU_Vector<T>::init()
{
    vlen = 0;
    offset = 0;
}

template <class T>
GPU_Vector<T>& GPU_Vector<T>::operator=(const GPU_Vector<T> &other)
{
    // check for self-assignment
    if(&other == this)
        return *this;

    // reuse storage when possible
    gpuarray.reset(new GPUArray(*(other.gpuarray)));
    vlen = other.vlen;
    return *this;
}

template <class T> GPU_Vector<T>::~GPU_Vector() { }

template struct GPU_Vector<int32_t>;
template struct GPU_Vector<float32_t>;
}
