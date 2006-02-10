#ifndef _SPARSEREALKERNEL_H___
#define _SPARSEREALKERNEL_H___

#include "kernel/SparseKernel.h"

class CSparseRealKernel : public CSparseKernel<REAL>
{
	public:
		CSparseRealKernel(LONG cachesize) : CSparseKernel<REAL>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSparseKernel<REAL>::init(l,r, do_init);

			ASSERT(l->get_feature_type()==F_REAL);
			ASSERT(r->get_feature_type()==F_REAL);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_REAL; }
};
#endif
