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
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VW_REGRESSOR_H__
#define _VW_REGRESSOR_H__

#include <shogun/classifier/vw/vw_environment.h>
#include <shogun/loss/LossFunction.h>

namespace shogun
{

/** @brief Regressor used by VW
 *
 * Stores the weight vectors and loss object, and is used for
 * calculating losses and updates.
 * 
 * The weight vector uses 'num_bits' number of bits, set in the
 * environment object to store weights.
 */
class VwRegressor
{
public:
	/**
	 * Default constructor, optionally taking an environment object
	 *
	 * @param env vw environment
	 */
	VwRegressor(VwEnvironment* env = NULL);

	/**
	 * Destructor
	 */
	~VwRegressor();

	/**
	 * Get loss for a label-prediction set
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return loss
	 */
	inline float64_t get_loss(float64_t prediction, float64_t label)
	{
		return loss->loss(prediction, label);
	}

	/**
	 * Get weight update for a prediction-label set
	 *
	 * @param prediction prediction
	 * @param label label
	 * @param eta_t learning rate
	 * @param norm scaling norm
	 *
	 * @return update
	 */
	inline float64_t get_update(float64_t prediction, float64_t label,
				    float64_t eta_t, float64_t norm)
	{
		return loss->get_update(prediction, label, eta_t, norm);
	}

	/**
	 * Initialize weight vectors
	 *
	 * @param env environment object
	 */
	void init(VwEnvironment* env = NULL);

public:
	/// Weight vectors, one array for each thread
	float** weight_vectors;
	/// Loss function
	CLossFunction* loss;
};

}
#endif // _VW_REGRESSOR_H__
