#ifndef _SHORTKERNEL_H___
#define _SHORTKERNEL_H___

#include "kernel/SimpleKernel.h"

class CShortKernel : public CSimpleKernel<SHORT>
{
	public:
		CShortKernel(long cachesize) : CSimpleKernel<SHORT>(cachesize)
		{
		}
		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_SHORT; }
};
#endif
