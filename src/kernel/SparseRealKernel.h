#ifndef _SPARSEREALKERNEL_H___
#define _SPARSEREALKERNEL_H___

#include "kernel/SparseKernel.h"

class CSparseRealKernel : public CSparseKernel<REAL>
{
	public:
		CSparseRealKernel(long cachesize) : CSparseKernel<REAL>(cachesize)
		{
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_REAL; }
};
#endif
