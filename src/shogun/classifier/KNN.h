/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef _KNN_H__
#define _KNN_H__

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
class CDistanceMachine;

/** @brief Class KNN, an implementation of the standard k-nearest neigbor
 * classifier.
 *
 * An example is classified to belong to the class of which the majority of the
 * k closest examples belong to. Formally, kNN is described as
 *
 * \f[
 * 		label for x = \arg \max_{l} \sum_{i=1}^{k} [label of i-th example = l]
 * \f]
 *
 * This class provides a capability to do weighted classfication using:
 *
 * \f[
 * 		label for x = \arg \max_{l} \sum_{i=1}^{k} [label of i-th example = l] q^{i},
 * \f]
 *
 * where \f$|q|<1\f$.
 *
 * To avoid ties, k should be an odd number. To define how close examples are
 * k-NN requires a CDistance object to work with (e.g., CEuclideanDistance ).
 *
 * Note that k-NN has zero training time but classification times increase
 * dramatically with the number of examples. Also note that k-NN is capable of
 * multi-class-classification. And finally, in case of k=1 classification will
 * take less time with an special optimization provided.
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

		/** classify all examples
		 *
		 * @return resulting labels
		 */
		virtual CLabels* apply();

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data);

		/// get output for example "vec_idx"
		virtual float64_t apply(int32_t vec_idx)
		{
			SG_ERROR( "for performance reasons use apply() instead of apply(int32_t vec_idx)\n");
			return 0;
		}

		/** classify all examples for 1...k
		 *
		 */
		SGMatrix<int32_t> classify_for_multiple_k();

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
		 * @param k k to be set
		 */
		inline void set_k(int32_t k)
		{
			ASSERT(k>0);
			m_k=k;
		}

		/** get k
		 *
		 * @return value of k
		 */
		inline int32_t get_k()
		{
			return m_k;
		}

		/** set q
		 * @param q value
		 */
		inline void set_q(float64_t q)
		{
			ASSERT(q<=1.0 && q>0.0);
			m_q = q;
		}

		/** get q
		 * @return q parameter
		 */
		inline float64_t get_q() { return m_q; }

		/* set whether to use cover trees for fast KNN
		 * @param use_covertree
		 */
		inline void set_use_covertree(bool use_covertree)
		{
			m_use_covertree = use_covertree;
		}

		/** get whether to use cover trees for fast KNN
		 * @return use_covertree parameter
		 */
		inline bool get_use_covertree() const { return m_use_covertree; }

		/** @return object name */
		inline virtual const char* get_name() const { return "KNN"; }

	protected:
		/** Stores feature data of underlying model.
		 *
		 * Replaces lhs and rhs of underlying distance with copies of themselves
		 */
		virtual void store_model_features();

		/** classify all examples with nearest neighbor (k=1)
		 * @return classified labels
		 */
		virtual CLabels* classify_NN();

		/** init distances to test examples
		 * @param data test examples
		 */
		void init_distance(CFeatures* data);

		/** train k-NN classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** constructs neighborhood matrix by distance
		 * @param N number of features
		 * @return matrix containing indexes of neighbors of i-th vector in 
		 * i-th column
		 */
		virtual SGMatrix<int32_t> get_neighborhood_matrix(int32_t N);

	private:
		void init();

	protected:
		/// the k parameter in KNN
		int32_t m_k;

		/// parameter q of rank weighting
		float64_t m_q;

		/// parameter to enable cover tree support
		bool m_use_covertree;

		///	number of classes (i.e. number of values labels can take)
		int32_t num_classes;

		///	smallest label, i.e. -1
		int32_t min_label;

		/** the actual trainlabels */
		SGVector<int32_t> train_labels;
};
}
#endif
