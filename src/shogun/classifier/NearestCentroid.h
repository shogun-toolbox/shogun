/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Philippe Tillet
 */

#ifndef _NEAREST_CENTROID_H__
#define _NEAREST_CENTROID_H__

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/CoverTree.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{

class CDistanceMachine;

/** @brief Class NearestCentroid, an implementation of Nearest Shrunk Centroid classifier
 * 
 * To define how close examples are
 * NearestCentroid requires a CDistance object to work with (e.g., CEuclideanDistance ).
 */

class CNearestCentroid : public CDistanceMachine{
	
public:
	/**
	 * Default constructor
	 */
	CNearestCentroid();
	
	/** constructor
	 *
	 * @param d distance
	 * @param trainlab labels for training
	 */
	CNearestCentroid(CDistance* distance, CLabels* trainlab);
	
	/** Destructor
	 */
	virtual ~CNearestCentroid();
	
	/** Set shrinking constant
	 * 
	 * @param shrinking to be set
	 */
	void set_shrinking(float64_t shrinking) {
		m_shrinking = shrinking ;
	}
	
	/** Get shrinking constant
	 * 
	 * @return value of the shrinking constant
	 */
	float64_t get_shrinking() const{
		return m_shrinking;
	}
	
	CSimpleFeatures<float64_t>* get_centroids() const{
		return m_centroids;
	}
	
protected:
	/** train Nearest Centroid classifier
	 *
	 * @param data training data (parameter can be avoided if distance or
	 * kernel-based classifiers are used and distance/kernels are
	 * initialized with train data)
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data=NULL);
	
	virtual void store_model_features();

private:
	void init();
	
protected:
	int32_t m_num_classes;
	float64_t m_shrinking;
	CSimpleFeatures<float64_t>* m_centroids;
	bool m_is_trained;
};

}

#endif