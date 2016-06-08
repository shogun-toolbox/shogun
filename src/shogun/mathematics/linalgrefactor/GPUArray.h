#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalgrefactor/GPUVector.h>
#include <memory>

#ifdef HAVE_VIENNACL
#include <viennacl/vector.hpp>
#endif

#ifndef GPU_ARRAY_H__
#define GPU_ARRAY_H__

namespace shogun
{

template <class T>
struct GPUVector<T>::GPUArray
{

#ifdef HAVE_VIENNACL
	typedef viennacl::backend::mem_handle VCLMemoryArray;
	typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;

	std::shared_ptr<VCLMemoryArray> GPUptr;
	index_t vlen;
	index_t offset;

	GPUArray(const SGVector<T> &vector);
	GPUArray(const GPUVector<T>::GPUArray &array);
	VCLVectorBase GPUvec();
	viennacl::const_entry_proxy<T> operator[](index_t index) const;
	viennacl::entry_proxy<T> operator[](index_t index);

#else //HAVE_VIENNACL
	GPUArray(const SGVector<T> &vector);
#endif //HAVE_VIENNACL

};

}

#endif

