#ifndef _SPARSEDREALKERNEL_H___
#define _SPARSEDREALKERNEL_H___

#include "kernel/SparseKernel.h"

class CSparseRealKernel : public CSparseKernel<DREAL>
{
	public:
		CSparseRealKernel(LONG cachesize) : CSparseKernel<DREAL>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSparseKernel<DREAL>::init(l,r, do_init);

			ASSERT(l->get_feature_type()==F_DREAL);
			ASSERT(r->get_feature_type()==F_DREAL);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }
};
#endif
