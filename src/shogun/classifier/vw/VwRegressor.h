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

#ifndef _VW_REGRESSOR_H__
#define _VW_REGRESSOR_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>
#include <shogun/classifier/vw/VwEnvironment.h>
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
class CVwRegressor: public CSGObject
{
public:
	/**
	 * Default constructor
	 */
	CVwRegressor();

	/**
	 * Constructor taking an environment object
	 *
	 * @param env_to_use environment
	 */
	CVwRegressor(CVwEnvironment* env_to_use);

	/**
	 * Destructor
	 */
	virtual ~CVwRegressor();

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
	 * Dump regressor in binary/text form
	 *
	 * @param reg_name output file name
	 * @param as_text whether to dump as text
	 */
	virtual void dump_regressor(char* reg_name, bool as_text);

	/**
	 * Load the regressor from a file
	 *
	 * @param file_name name of dumped regressor binary file
	 */
	virtual void load_regressor(char* file_name);

	/**
	 * Return name of the object
	 * @return VwRegressor
	 */
	virtual const char* get_name() const { return "VwRegressor"; }

	/**
	 * Initialize weight vectors
	 *
	 * @param env_to_use environment object
	 */
	virtual void init(CVwEnvironment* env_to_use = NULL);

public:
	/// Weight vectors, one array for each thread
	float32_t** weight_vectors;
	/// Loss function
	CLossFunction* loss;

protected:
	/// Environment
	CVwEnvironment* env;
};

}
#endif // _VW_REGRESSOR_H__
