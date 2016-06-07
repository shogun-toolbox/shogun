#include <shogun/mathematics/linalgrefactor/GPUBackend.h>
#include <shogun/mathematics/linalgrefactor/GPUArray.h>

#ifdef HAVE_VIENNACL
#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#endif

namespace shogun
{
/** GPU calculation backend **/
template <typename T> T GPUBackend::dot(const GPU_Vector<T> &a, const GPU_Vector<T> &b)
{
#ifdef HAVE_VIENNACL
    /**
    * Method that computes the dot product using ViennaCL
    */
    //return T(0);
    return viennacl::linalg::inner_prod(a.gpuarray->GPUvec(), b.gpuarray->GPUvec());
// similarly, other methods
#else
    SG_SERROR("User did not register GPU backend. \n");
    return T(0);
#endif
}

template int32_t GPUBackend::dot<int32_t>(const GPU_Vector<int32_t> &a, const GPU_Vector<int32_t> &b);
template float32_t GPUBackend::dot<float32_t>(const GPU_Vector<float32_t> &a, const GPU_Vector<float32_t> &b);
}
