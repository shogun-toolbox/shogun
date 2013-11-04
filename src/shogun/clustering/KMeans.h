/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KMEANS_H__
#define _KMEANS_H__

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
class CDistanceMachine;

/** @brief KMeans clustering,  partitions the data into k (a-priori specified) clusters.
 *
 * It minimizes
 * \f[
 *  \sum_{i=1}^k\sum_{x_j\in S_i} (x_j-\mu_i)^2
 * \f]
 *
 * where \f$\mu_i\f$ are the cluster centers and \f$S_i,\;i=1,\dots,k\f$ are the index
 * sets of the clusters.
 *
 * Beware that this algorithm obtains only a <em>local</em> optimum.
 *
 * cf. http://en.wikipedia.org/wiki/K-means_algorithm */
class CKMeans : public CDistanceMachine
{
	public:
		/** default constructor */
		CKMeans();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 */
		CKMeans(int32_t k, CDistance* d);

		/** constructor for supplying initial centers
		 * @param k_i parameter k
		 * @param d_i distance
		 * @param centers_i initial centers for KMeans algorithm
		*/
		CKMeans(int32_t k_i, CDistance* d_i, float64_t* centers_i );
		virtual ~CKMeans();


		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** get classifier type
		 *
		 * @return classifier type KMEANS
		 */
		virtual EMachineType get_classifier_type() { return CT_KMEANS; }

		/** load distance machine from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save distance machine to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** set k
		 *
		 * @param p_k new k
		 */
		void set_k(int32_t p_k);

		/** get k
		 *
		 * @return the parameter k
		 */
		int32_t get_k();

		/** set maximum number of iterations
		 *
		 * @param iter the new maximum
		 */
		void set_max_iter(int32_t iter);

		/** get maximum number of iterations
		 *
		 * @return maximum number of iterations
		 */
		float64_t get_max_iter();

		/** get radiuses
		 *
		 * @return radiuses
		 */
		SGVector<float64_t> get_radiuses();

		/** get centers
		 *
		 * @return cluster centers or empty matrix if no radiuses are there (not trained yet)
		 */
		SGMatrix<float64_t> get_cluster_centers();

		/** get dimensions
		 *
		 * @return number of dimensions
		 */
		int32_t get_dimensions();

		/** @return object name */
		virtual const char* get_name() const { return "KMeans"; }

		/* get the initial centers */
		virtual bool set_initial_centers(float64_t* centers);

	protected:
		/** clustknb
		 *
		 * @param use_old_mus if old mus shall be used
		 * @param mus_start mus start
		 */
		void clustknb(bool use_old_mus, float64_t *mus_start);

		/** train k-means
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** Ensures cluster centers are in lhs of underlying distance */
		virtual void store_model_features();

		virtual bool train_require_labels() const { return false; }

	private:
		void init();

	protected:
		/// maximum number of iterations
		int32_t max_iter;

		/// the k parameter in KMeans
		int32_t k;

		/// number of dimensions
		int32_t dimensions;

		/// radi of the clusters (size k)
		SGVector<float64_t> R;

		///initial centers supplied
		float64_t* mus_initial;
	private:
		/* temporary variable for weighting over the train data */
		SGVector<float64_t> Weights;

		/* temp variable for cluster centers */
		SGMatrix<float64_t> mus;

		/* flag to indicate if initial centers have been supplied */
		bool mus_set;
};
}
#endif

