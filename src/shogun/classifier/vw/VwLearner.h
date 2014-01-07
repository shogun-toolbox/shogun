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

#ifndef _VW_LEARNER_H__
#define _VW_LEARNER_H__

#include <base/SGObject.h>
#include <base/Parameter.h>
#include <classifier/vw/vw_common.h>
#include <classifier/vw/VwRegressor.h>

namespace shogun
{
/** @brief Base class for all VW learners
 *
 * Learners are supplied with a regressor and the environment.
 *
 * They should implement a train function which updates
 * the weight vector given the update for the example.
 */
class CVwLearner: public CSGObject
{
public:
	/**
	 * Default constructor
	 */
	CVwLearner()
		: CSGObject(), reg(NULL), env(NULL)
	{
		register_learner_params();
	}

	/**
	 * Constructor, initializes regressor and environment
	 *
	 * @param regressor regressor
	 * @param vw_env environment
	 */
	CVwLearner(CVwRegressor* regressor, CVwEnvironment* vw_env)
		: CSGObject(), reg(regressor), env(vw_env)
	{
		SG_REF(reg);
		SG_REF(env);
		register_learner_params();
	}

	/**
	 * Destructor
	 */
	virtual ~CVwLearner()
	{
		if (reg)
			SG_UNREF(reg);
		if (env)
			SG_UNREF(env);
	}

	/**
	 * Add parameters to make them serializable
	 */
	void register_learner_params()
	{
		SG_ADD((CSGObject**) &reg, "vw_regressor", "Regressor object",
		       MS_NOT_AVAILABLE);
		SG_ADD((CSGObject**) &env, "vw_env", "Environment",
		       MS_NOT_AVAILABLE);
	}

	/**
	 * Train on the example
	 *
	 * @param ex example
	 * @param update update
	 */
	virtual void train(VwExample* &ex, float32_t update) = 0;

	/**
	 * Return the name of the object
	 *
	 * @return VwLearner
	 */
	virtual const char* get_name() const { return "VwLearner"; }

protected:
	/// Regressor object that will be used for getting updates
	CVwRegressor *reg;
	/// Environment
	CVwEnvironment *env;
};
}
#endif // _VW_LEARNER_H__
