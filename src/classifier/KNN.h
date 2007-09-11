/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KNN_H__
#define _KNN_H__

#include <stdio.h>
#include "lib/common.h"
#include "lib/io.h"
#include "features/Features.h"
#include "distance/Distance.h"
#include "distance/DistanceMachine.h"

class CDistanceMachine;

class CKNN : public CDistanceMachine
{
	public:
		CKNN();
		virtual ~CKNN();

		virtual inline EClassifierType get_classifier_type() { return CT_KNN; }
		//inline EDistanceType get_distance_type() { return DT_KNN;}
		virtual bool train();
		virtual CLabels* classify(CLabels* output=NULL);
		virtual DREAL classify_example(INT idx)
		{
			SG_ERROR( "for performance reasons use classify() instead of classify_example\n");
			return 0;
		}

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline void set_k(DREAL p_k) 
		{
			ASSERT(p_k>0);
			this->k=p_k;
		}

		inline DREAL get_k()
		{
			return k;
		}

	protected:
		/// the k parameter in KNN
		DREAL k;

		///	number of classes (i.e. number of values labels can take)
		int num_classes;

		///	smallest label, i.e. -1
		int min_label;

		/// number of train examples
		int num_train_labels;

		/// the actual trainlabels
		INT* train_labels;
};
#endif

