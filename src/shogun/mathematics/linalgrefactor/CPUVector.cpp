#include <shogun/mathematics/linalgrefactor/CPUVector.h>

namespace shogun

{
template <class T>
CPUVector<T>::CPUVector():CPUptr(nullptr), vlen(0) { }

template <class T>
CPUVector<T>::CPUVector(const SGVector<T> &vector)
: CPUptr(vector.vector), vlen(vector.vlen) { }

template <class T>
CPUVector<T>::CPUVector(const CPUVector<T> &vector)
: CPUptr(vector.CPUptr), vlen(vector.vlen) { }

template struct CPUVector<int32_t>;
template struct CPUVector<float32_t>;
}
