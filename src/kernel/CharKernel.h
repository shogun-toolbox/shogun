#ifndef _CHARKERNEL_H___
#define _CHARKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/common.h"

class CCharKernel : public CSimpleKernel<CHAR>
{
	public:
		CCharKernel(LONG cachesize) : CSimpleKernel<CHAR>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleKernel<CHAR>::init(l,r, do_init);

			assert(l->get_feature_type()==F_CHAR);
			assert(r->get_feature_type()==F_CHAR);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_CHAR; }
};
#endif
