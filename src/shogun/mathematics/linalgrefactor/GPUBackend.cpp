#include <shogun/mathematics/linalgrefactor/GPUBackend.h>
#include <shogun/mathematics/linalgrefactor/GPUArray.h>

#ifdef HAVE_VIENNACL
#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#endif

namespace shogun
{

template <typename T>
T GPUBackend::dot(const GPUVector<T> &a, const GPUVector<T> &b) const
{
#ifdef HAVE_VIENNACL
	/**
	* Method that computes the dot product using ViennaCL
	*/
    	return viennacl::linalg::inner_prod(a.gpuarray->GPUvec(), b.gpuarray->GPUvec());

#else
	SG_SERROR("User did not register GPU backend. \n");
	return T(0);
#endif
}

template int32_t GPUBackend::dot<int32_t>(const GPUVector<int32_t> &a, const GPUVector<int32_t> &b) const;
template float32_t GPUBackend::dot<float32_t>(const GPUVector<float32_t> &a, const GPUVector<float32_t> &b) const;

}
