/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
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

/** class KNN */
class CKNN : public CDistanceMachine
{
	public:
		/** default constructor */
		CKNN();

		/** constructor
		 *
		 * @param k k
		 * @param d distance
		 * @param trainlab labels for training
		 */
		CKNN(int32_t k, CDistance* d, CLabels* trainlab);
		virtual ~CKNN();

		/** get classifier type
		 *
		 * @return classifier type KNN
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_KNN; }
		//inline EDistanceType get_distance_type() { return DT_KNN;}

		/** train classifier
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/** classify all examples
		 *
		 * @param output resulting labels
		 * @return resulting labels
		 */
		virtual CLabels* classify(CLabels* output=NULL);

		/// get output for example "vec_idx"
		virtual DREAL classify_example(int32_t vec_idx)
		{
			SG_ERROR( "for performance reasons use classify() instead of classify_example\n");
			return 0;
		}

		/** load from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** set k
		 *
		 * @param p_k new k
		 */
		inline void set_k(DREAL p_k)
		{
			ASSERT(p_k>0);
			this->k=p_k;
		}

		/** get k
		 *
		 * @return k
		 */
		inline DREAL get_k()
		{
			return k;
		}

	protected:
		/// the k parameter in KNN
		DREAL k;

		///	number of classes (i.e. number of values labels can take)
		int32_t num_classes;

		///	smallest label, i.e. -1
		int32_t min_label;

		/// number of train examples
		int32_t num_train_labels;

		/// the actual trainlabels
		int32_t* train_labels;
};
#endif

