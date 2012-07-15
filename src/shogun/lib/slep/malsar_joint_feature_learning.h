/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Jiayu Zhou and Jieping Ye
 */

#ifndef  MALSAR_JOINT_FEATURE_LEARNING_H_
#define  MALSAR_JOINT_FEATURE_LEARNING_H_
#include <shogun/lib/slep/slep_options.h>
#include <shogun/features/DotFeatures.h>

namespace shogun 
{

slep_result_t malsar_joint_feature_learning(
		CDotFeatures* features,
		double* y,
		double rho1,
		double rho2,
		const slep_options& options);

};
#endif   /* ----- #ifndef MALSAR_JOINT_FEATURE_LEARNING_H_  ----- */

