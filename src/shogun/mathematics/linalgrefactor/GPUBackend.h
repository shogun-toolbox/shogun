#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalgrefactor/BaseVector.h>
#include <shogun/mathematics/linalgrefactor/GPUVector.h>
#include <memory>

#ifndef GPUBACKEND_H__
#define GPUBACKEND_H__

namespace shogun
{

class GPUBackend
{

template <class T> friend struct GPUVector;

public:
	template <typename T>
	T dot(const GPUVector<T> &a, const GPUVector<T> &b) const;
};

}

#endif
