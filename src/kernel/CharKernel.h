#ifndef _CHARKERNEL_H___
#define _CHARKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/io.h"

class CCharKernel : public CSimpleKernel<CHAR>
{
	public:
		CCharKernel(long cachesize) : CSimpleKernel<CHAR>(cachesize)
		{
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_CHAR; }
};
#endif
