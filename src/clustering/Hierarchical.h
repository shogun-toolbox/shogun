/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HIERARCHICAL_H__
#define _HIERARCHICAL_H__

#include <stdio.h>
#include "lib/common.h"
#include "lib/io.h"
#include "features/RealFeatures.h"
#include "distance/Distance.h"
#include "distance/DistanceMachine.h"

class CDistanceMachine;

class CHierarchical : public CDistanceMachine
{
	public:
		CHierarchical();
		virtual ~CHierarchical();

		virtual inline EClassifierType get_classifier_type() { return CT_HIERARCHICAL; }

		virtual bool train();
		virtual CLabels* classify(CLabels* output=NULL);
		virtual DREAL classify_example(INT idx)
		{
			SG_ERROR( "for performance reasons use classify() instead of classify_example\n");
			return 0;
		}

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline void set_k(INT p_k) 
		{
			ASSERT(p_k>0);
			this->k=p_k;
		}

		inline DREAL get_k()
		{
			return k;
		}

		inline void get_assignments(DREAL*& assign, INT& rows, INT& num)
		{
			assign=assignment;
			rows=2;
			num=k;
		}

		inline void get_pairs(INT*& tuples, INT& rows, INT& num)
		{
			tuples=pairs;
			rows=2;
			num=k;
		}

	protected:
		/// the k parameter in Hierarchical
		INT k;

		/// number of dimensions
		INT dimensions;

		/// radi of the clusters (size k)
		DREAL* assignment;
		
		/// tuples of i/j
		INT* pairs;
};
#endif
