#ifndef _SHORTKERNEL_H___
#define _SHORTKERNEL_H___

#include "kernel/SimpleKernel.h"

class CShortKernel : public CSimpleKernel<SHORT>
{
	public:
		CShortKernel(long cachesize) : CSimpleKernel<SHORT>(cachesize)
		{
		}
};
#endif
