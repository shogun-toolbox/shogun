/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DISTANCE_MACHINE_H__
#define _DISTANCE_MACHINE_H__

#include <lib/common.h>
#include <distance/Distance.h>
#include <labels/Labels.h>
#include <labels/RegressionLabels.h>
#include <machine/Machine.h>

#include <stdio.h>

namespace shogun
{
	class CLabels;
	class CRegressionLabels;
	class CDistance;
	class CMachine;

/** @brief A generic DistanceMachine interface.
 *
 * A distance machine is based on a a-priori choosen distance.
 */
class CDistanceMachine : public CMachine
{
	public:
		/** default constructor */
		CDistanceMachine();

		/** destructor */
		virtual ~CDistanceMachine();

		/** set distance
		 *
		 * @param d distance to set
		 */
		void set_distance(CDistance* d);

		/** get distance
		 *
		 * @return distance
		 */
		CDistance* get_distance() const;

		/**
		 * get distance functions for lhs feature vectors
		 * going from a1 to a2 and rhs feature vector b
		 *
		 * @param result array of distance values
		 * @param idx_a1 first feature vector a1 at idx_a1
		 * @param idx_a2 last feature vector a2 at idx_a2
		 * @param idx_b feature vector b at idx_b
		 */
		void distances_lhs(float64_t* result,int32_t idx_a1,int32_t idx_a2,int32_t idx_b);

		/**
		 * get distance functions for rhs feature vectors
		 * going from b1 to b2 and lhs feature vector a
		 *
		 * @param result array of distance values
		 * @param idx_b1 first feature vector a1 at idx_b1
		 * @param idx_b2 last feature vector a2 at idx_b2
		 * @param idx_a feature vector a at idx_a
		 */
		void distances_rhs(float64_t* result,int32_t idx_b1,int32_t idx_b2,int32_t idx_a);

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "DistanceMachine"; }

		/** Classify all provided features.
		 * Cluster index with smallest distance to to be classified element is
		 * returned
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

		/** Apply machine to one example.
		 * Cluster index with smallest distance to to be classified element is
		 * returned
		 *
		 * @param num which example to apply machine to
		 * @return cluster label nearest to example
		 */
		virtual float64_t apply_one(int32_t num);

	protected:
		/** Ensures cluster centers are in lhs of underlying distance
		 *
		 * NOT IMPLEMENTED!
		 * Base method. Is called automatically after train because flag is
		 * always true for distance machines.
		 * Since every distance machine has to make sure that
		 * cluster centers are in lhs of distance variable, it is unimplemented
		 * here and HAS to be implemented in subclasses.
		 */
		virtual void store_model_features();

		/**
		 * thread function for computing distance values
		 *
		 * @param p thread parameter
		 */
		static void* run_distance_thread_lhs(void* p);

		/**
		 * thread function for computing distance values
		 *
		 * @param p thread parameter
		 */
		static void* run_distance_thread_rhs(void* p);

	private:
		void init();

	protected:
		/** the distance */
		CDistance* distance;
};
}
#endif
