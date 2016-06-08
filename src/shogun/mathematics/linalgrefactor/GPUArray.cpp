#include <shogun/mathematics/linalgrefactor/GPUArray.h>

namespace shogun
{

#ifdef HAVE_VIENNACL

template <class T>
GPUVector<T>::GPUArray::GPUArray(const SGVector<T> &vector)
:GPUptr(new VCLMemoryArray()), vlen(vector.vlen), offset(0)
{
	viennacl::backend::memory_create(*GPUptr, sizeof(T)*vlen,
        	viennacl::context());

	viennacl::backend::memory_write(*GPUptr, 0, vlen*sizeof(T),
        	vector.vector);
}

template <class T>
GPUVector<T>::GPUArray::GPUArray(const GPUVector<T>::GPUArray &array)
{
	GPUptr = array.GPUptr;
	vlen = array.vlen;
	offset = array.offset;
}

template <class T>
typename GPUVector<T>::GPUArray::VCLVectorBase GPUVector<T>::GPUArray::GPUvec()
{
        return VCLVectorBase(*GPUptr, vlen, offset, 1);
}

template <class T>
typename viennacl::const_entry_proxy<T>
GPUVector<T>::GPUArray::operator[](index_t index) const
{
	return viennacl::const_entry_proxy<T>(offset+index, *GPUptr);
}

template <class T>
typename viennacl::entry_proxy<T>
GPUVector<T>::GPUArray::operator[](index_t index)
{
	return viennacl::entry_proxy<T>(offset+index, *GPUptr);
}

#else // HAVE_VIENNACL

template <class T>
GPUVector<T>::GPUArray::GPUArray(const SGVector<T> &vector)
{
	SG_SERROR("User did not register GPU backend. \n");
}

#endif //HAVE_VIENNACL

template struct GPUVector<int32_t>;
template struct GPUVector<float32_t>;

}
