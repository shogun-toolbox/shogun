/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VW_NONADAPTIVE_H__
#define _VW_NONADAPTIVE_H__

#include <shogun/lib/config.h>
#include <shogun/classifier/vw/VwLearner.h>
#include <shogun/classifier/vw/vw_common.h>

namespace shogun
{
/** @brief VwNonAdaptiveLearner uses a standard gradient descent
 * weight update rule.
 *
 * The weight vector is updated by adding the corresponding feature
 * multiplied by the update factor for the example.
 */
class CVwNonAdaptiveLearner: public CVwLearner
{
public:
	/**
	 * Default constructor
	 */
	CVwNonAdaptiveLearner();

	/**
	 * Constructor, initializes regressor and environment
	 *
	 * @param regressor regressor to use
	 * @param vw_env environment to use
	 */
	CVwNonAdaptiveLearner(CVwRegressor* regressor, CVwEnvironment* vw_env);

	/**
	 * Destructor
	 */
	virtual ~CVwNonAdaptiveLearner();

	/**
	 * Train on one example, given the update
	 *
	 * @param ex example
	 * @param update the update
	 */
	virtual void train(VwExample* &ex, float32_t update);

	/**
	 * Return the name of the object
	 *
	 * @return VwNonAdaptiveLearner
	 */
	virtual const char* get_name() const { return "VwNonAdaptiveLearner"; }

private:
	/**
	 * Perform the update for paired features
	 *
	 * @param weights weights
	 * @param page_feature feature belonging to current namespace
	 * @param offer_features features belonging to paired namespace
	 * @param mask mask
	 * @param update update
	 */
	void quad_update(float32_t* weights, VwFeature& page_feature,
			 v_array<VwFeature> &offer_features, vw_size_t mask,
			 float32_t update);
};
}
#endif // _VW_NONADAPTIVE_H__
