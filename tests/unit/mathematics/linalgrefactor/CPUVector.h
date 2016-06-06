#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalgrefactor/BaseVector.h>

#ifndef CPUVECTOR_H__
#define CPUVECTOR_H__

namespace shogun
{

template <class T>
struct CPUVector : public BaseVector<T>
{
    T* CPUptr;
    index_t vlen;

    CPUVector();

    CPUVector(const SGVector<T> &vector);

    CPUVector(const CPUVector<T> &vector);

    bool onGPU() { return false; }
};

}

#endif
