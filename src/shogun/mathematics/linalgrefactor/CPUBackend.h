#include <shogun/lib/config.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalgrefactor/CPUVector.h>

#ifndef CPUBACKEND_H__
#define CPUBACKEND_H__

namespace shogun
{

/** Backend Class **/
class CPUBackend
{
public:

	CPUBackend();
	CPUBackend(const CPUBackend& cpubackend);

	template <typename T>
	T dot(const CPUVector<T> &a, const CPUVector<T> &b) const;
};

}

#endif
