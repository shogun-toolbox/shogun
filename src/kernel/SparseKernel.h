/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEKERNEL_H___
#define _SPARSEKERNEL_H___

#include "kernel/Kernel.h"
#include "features/SparseFeatures.h"

/** template class SparseKernel */
template <class ST> class CSparseKernel : public CKernel
{
	public:
		/** constructor
		 *
		 * @param cachesize cache size
		 */
		CSparseKernel(INT cachesize) : CKernel(cachesize) {}

		/** constructor
		 *
		 * @param l features for left-hand side
		 * @param r features for right-hand side
		 */
		CSparseKernel(CFeatures* l, CFeatures* r) : CKernel(10)
		{
			init(l, r);
		}

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CKernel::init(l,r);

			ASSERT(l->get_feature_class()==C_SPARSE);
			ASSERT(r->get_feature_class()==C_SPARSE);
			ASSERT(l->get_feature_type()==this->get_feature_type());
			ASSERT(r->get_feature_type()==this->get_feature_type());

			if (((CSparseFeatures<ST>*) lhs)->get_num_features() != ((CSparseFeatures<ST>*) rhs)->get_num_features())
			{
				SG_ERROR( "train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						((CSparseFeatures<ST>*) lhs)->get_num_features(),((CSparseFeatures<ST>*)rhs)->get_num_features());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SPARSE
		 */
		inline virtual EFeatureClass get_feature_class() { return C_SPARSE; }

		/** return feature type the kernel can deal with
		 *
		 * @return templated feature type
		 */
		inline virtual EFeatureType get_feature_type();
};

template<> inline EFeatureType CSparseKernel<DREAL>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CSparseKernel<ULONG>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CSparseKernel<INT>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CSparseKernel<WORD>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CSparseKernel<SHORT>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CSparseKernel<BYTE>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CSparseKernel<CHAR>::get_feature_type() { return F_CHAR; }

#endif /* _SPARSEKERNEL_H__ */
