/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SIMPLEKERNEL_H___
#define _SIMPLEKERNEL_H___

#include "kernel/Kernel.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

/** Template class SimpleKernel is the base class for kernels working on dense
 * features, i.e. they all derive from this class (cf. e.g. CGaussianKernel)
 */ 
template <class ST> class CSimpleKernel : public CKernel
{
	public:
		/** constructor
		 *
		 * @param cachesize cache size
		 */
		CSimpleKernel(int32_t cachesize) : CKernel(cachesize) {}

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CSimpleKernel(CFeatures* l, CFeatures* r) : CKernel(10)
		{
			init(l, r);
		}

		/** initialize kernel
		 *  e.g. setup lhs/rhs of kernel, precompute normalization
		 *  constants etc.
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *
		 *  @param l features for left-hand side
		 *  @param r features for right-hand side
		 *  @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CKernel::init(l,r);

			ASSERT(l->get_feature_class()==C_SIMPLE);
			ASSERT(r->get_feature_class()==C_SIMPLE);
			ASSERT(l->get_feature_type()==this->get_feature_type());
			ASSERT(r->get_feature_type()==this->get_feature_type());

			if ( ((CSimpleFeatures<ST>*) l)->get_num_features() != ((CSimpleFeatures<ST>*) r)->get_num_features() )
			{  
				SG_ERROR( "train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						((CSimpleFeatures<ST>*) l)->get_num_features(),((CSimpleFeatures<ST>*) r)->get_num_features());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SIMPLE
		 */
		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }

		/** return feature type the kernel can deal with
		 *
		 * @return templated feature type
		 */
		inline virtual EFeatureType get_feature_type();
};


template<> inline EFeatureType CSimpleKernel<DREAL>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CSimpleKernel<SHORTREAL>::get_feature_type() { return F_SHORTREAL; }

template<> inline EFeatureType CSimpleKernel<uint64_t>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CSimpleKernel<int32_t>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CSimpleKernel<uint16_t>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CSimpleKernel<SHORT>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CSimpleKernel<uint8_t>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CSimpleKernel<char>::get_feature_type() { return F_CHAR; }


#endif /* _SIMPLEKERNEL_H__ */
