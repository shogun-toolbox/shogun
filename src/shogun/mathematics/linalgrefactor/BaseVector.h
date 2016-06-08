#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>

#ifndef BASEVECTOR_H__
#define BASEVECTOR_H__

namespace shogun
{

/** Vector structure **/
template <class T>
struct BaseVector
{
	BaseVector()
	{
	}
    
	virtual bool onGPU() = 0 ;

	index_t vlen;
};
}

#endif
