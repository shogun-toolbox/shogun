#ifndef _REALKERNEL_H___
#define _REALKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/common.h"

class CRealKernel : public CSimpleKernel<REAL>
{
	public:
		CRealKernel(long cachesize) : CSimpleKernel<REAL>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleKernel<REAL>::init(l,r, do_init);

			assert(l->get_feature_type()==F_REAL);
			assert(r->get_feature_type()==F_REAL);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_REAL; }
};
#endif
