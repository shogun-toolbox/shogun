#include <shogun/mathematics/linalgrefactor/CPUBackend.h>

namespace shogun
{

template <typename T> T CPUBackend::dot(const CPUVector<T> &a, const CPUVector<T> &b)
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
    Eigen::Map<VectorXt> vec_a(a.CPUptr, a.vlen);
    Eigen::Map<VectorXt> vec_b(b.CPUptr, b.vlen);
    return vec_a.dot(vec_b);
}

template int32_t CPUBackend::dot<int32_t>(const CPUVector<int32_t> &a, const CPUVector<int32_t> &b);
}
