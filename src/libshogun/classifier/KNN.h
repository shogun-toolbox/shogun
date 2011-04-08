/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KNN_H__
#define _KNN_H__

#include <stdio.h>
#include "lib/common.h"
#include "lib/io.h"
#include "features/Features.h"
#include "distance/Distance.h"
#include "classifier/DistanceMachine.h"

namespace shogun
{
class CDistanceMachine;

/** @brief Class KNN, an implementation of the standard k-nearest neigbor
 * classifier.
 *
 * An example is classified to belong to the class of which the majority of the
 * k closest examples belong to.
 *
 * To avoid ties, k should be an odd number. To define how close examples are
 * k-NN requires a CDistance object to work with (e.g., CEuclideanDistance ).
 *
 * Note that k-NN has zero training time but classification times increase
 * dramatically with the number of examples. Also note that k-NN is capable of
 * multi-class-classification.
 */
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

		/** train k-NN classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** classify all examples
		 *
		 * @return resulting labels
		 */
		virtual CLabels* classify();

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* classify(CFeatures* data);

		/// get output for example "vec_idx"
		virtual float64_t classify_example(int32_t vec_idx)
		{
			SG_ERROR( "for performance reasons use classify() instead of classify_example\n");
			return 0;
		}

		/** classify all examples for 1...k
		 *
		 * @param output resulting labels for all k
		 * @param k_out number of columns (k)
		 * @param num_vec number of outputs
		 */
		void classify_for_multiple_k(int32_t** output, int32_t* num_vec, int32_t* k_out);

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
		inline void set_k(int32_t p_k)
		{
			ASSERT(p_k>0);
			this->k=p_k;
		}

		/** get k
		 *
		 * @return k
		 */
		inline int32_t get_k()
		{
			return k;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "KNN"; }

	protected:
		/** classify all examples with nearest neighbor (k=1)
		 * @return classified labels
		 */
		virtual CLabels* classify_NN();

		/** init distances to test examples
		 * @param test examples
		 */
		void init_distance(CFeatures* data);

	protected:
		/// the k parameter in KNN
		int32_t k;

		///	number of classes (i.e. number of values labels can take)
		int32_t num_classes;

		///	smallest label, i.e. -1
		int32_t min_label;

		/// number of train examples
		int32_t num_train_labels;

		/// the actual trainlabels
		int32_t* train_labels;
};
}
#endif
