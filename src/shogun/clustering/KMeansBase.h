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

#ifndef _KMEANSBASE_H__
#define _KMEANSBASE_H__

#include <shogun/lib/config.h>

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
class CDistanceMachine;

/** @brief Class KMeansBase, a base class for various KMeans implementations.
 * 
 * The train_machine method should be overridden for different KMeans implementations.
 */
class CKMeansBase : public CDistanceMachine
{
	public:
		/** default constructor */
		CKMeansBase();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 * @param kmeanspp true for using KMeans++ (default false)
		 */
		CKMeansBase(int32_t k, CDistance* d, bool kmeanspp=false);

		/** constructor for supplying initial centers
		 * @param k_i parameter k
		 * @param d_i distance
		 * @param centers_i initial centers for KMeans algorithm
		*/
		CKMeansBase(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i);
		virtual ~CKMeansBase();


		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** get classifier type
		 *
		 * @return classifier type KMEANS
		 */
		virtual EMachineType get_classifier_type() { return CT_KMEANS; }

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

		/** set use_kmeanspp attribute
		 *
		 * @param kmpp true=>use KMeans++ false=>don't use KMeans++
		 */
		void set_use_kmeanspp(bool kmpp);

		/** get use_kmeanspp attribute
		 *
		 * @return use_kmeanspp true=>use KMeans++ false=>don't use KMeans++
		 */
		bool get_use_kmeanspp() const;

		/** set fixed centers
		 *
		 * @param fixed true if fixed cluster centers are intended
		 */
		void set_fixed_centers(bool fixed);

		/** get fixed centers
		 *
		 * @return whether boolean centers are to be used
		 */
		bool get_fixed_centers();

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
		virtual const char* get_name() const { return "KMeansBase"; }

		/** set the initial cluster centers 
		 *
		 * @param centers matrix with cluster centers (k colums, dim rows)
		 */
		virtual void set_initial_centers(SGMatrix<float64_t> centers);
		
	private:
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

		void init();


	protected:

		/// maximum number of iterations
		int32_t max_iter;

		/// whether to keep cluster centers fixed or not
		bool fixed_centers;

		/// the k parameter in KMeans
		int32_t k;

		/// number of dimensions
		int32_t dimensions;

		/// radi of the clusters (size k)
		SGVector<float64_t> R;

		///initial centers supplied
		SGMatrix<float64_t> m_mus_initial;
		
		///flag to check if kmeans++ has to be used
		bool use_kmeanspp;
	
		/// temp variable for cluster centers
		SGMatrix<float64_t> m_mus;

	protected:

		/** kmeans++ algorithm to initialize cluster centers
		* 
		* @return initial cluster centers: matrix (k columns, dim rows)
		*/	
		SGMatrix<float64_t> kmeanspp();

		/** initialize for training
		*/
		int32_t initialize_training(CFeatures* data);

		/** algorithm to initialize random cluster centers
		* 
		* @return initial cluster centers: matrix (k columns, dim rows)
		*/
		void set_random_centers(SGVector<float64_t> weights_set, SGVector<int32_t> cl_list, int32_t XSize);
		void set_initial_centers(SGVector<float64_t> weights_set, 
					SGVector<int32_t> cl_list, int32_t XSize);
		void compute_cluster_variances();

};
}
#endif

