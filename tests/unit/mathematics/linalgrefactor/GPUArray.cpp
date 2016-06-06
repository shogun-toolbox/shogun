#include <shogun/mathematics/linalgrefactor/GPUArray.h>

namespace shogun
{

#ifdef HAVE_VIENNACL

template <class T>
GPU_Vector<T>::GPUArray::GPUArray(const SGVector<T> &vector)
:GPUptr(new VCLMemoryArray()), vlen(vector.vlen), offset(0)
{
    viennacl::backend::memory_create(*GPUptr, sizeof(T)*vlen,
        viennacl::context());

    viennacl::backend::memory_write(*GPUptr, 0, vlen*sizeof(T),
        vector.vector);
}

template <class T>
GPU_Vector<T>::GPUArray::GPUArray(const GPU_Vector<T>::GPUArray &array)
//:GPUptr(new VCLMemoryArray()), vlen(array.vlen), offset(0)
{
    GPUptr = array.GPUptr;
	vlen = array.vlen;
	offset = array.offset;
}

template <class T>
typename GPU_Vector<T>::GPUArray::VCLVectorBase GPU_Vector<T>::GPUArray::GPUvec()
{
        return VCLVectorBase(*GPUptr, vlen, offset, 1);
}

#else

template <class T>
GPU_Vector<T>::GPUArray::GPUArray(const SGVector<T> &vector)
{
    SG_SERROR("User did not register GPU backend. \n");
}

#endif

template struct GPU_Vector<int32_t>;

}
