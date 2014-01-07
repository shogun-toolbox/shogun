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
#include <lib/common.h>
#include <io/SGIO.h>
#include <features/Features.h>
#include <features/DenseFeatures.h>
#include <distance/Distance.h>
#include <machine/DistanceMachine.h>

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

	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_MULTICLASS);

	/**
	 * Default constructor
	 */
	CNearestCentroid();

	/** constructor
	 *
	 * @param distance distance
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

	/** Get the centroids
	 *
	 * @return Matrix containing the centroids
	 */
	CDenseFeatures<float64_t>* get_centroids() const{
		return m_centroids;
	}

	/** Returns the name of the SGSerializable instance.
	 *
	 * @return name of the SGSerializable
	 */
	virtual const char* get_name() const { return "NearestCentroid"; }

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

	/** Stores feature data of underlying model.
	 *
	 * Sets centroids as lhs
	 */

private:
	void init();

protected:
	///	number of classes (i.e. number of values labels can take)
	int32_t m_num_classes;

	///	Shrinking parameter
	float64_t m_shrinking;

	///	The centroids of the trained features
	CDenseFeatures<float64_t>* m_centroids;

	///	Tells if the classifier has been trained or not
	bool m_is_trained;
};

}

#endif
