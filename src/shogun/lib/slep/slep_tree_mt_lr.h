/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2010-2012 Jun Liu, Jieping Ye 
 */

#ifndef  SLEP_TREE_MT_LR_H_
#define  SLEP_TREE_MT_LR_H_

#include <shogun/lib/slep/slep_options.h>
#include <shogun/features/DotFeatures.h>

namespace shogun 
{

slep_result_t slep_tree_mt_lr(
		CDotFeatures* features,
		double* y,
		double z,
		const slep_options& options);

};
#endif   /* ----- #ifndef SLEP_TREE_MT_LR_H_  ----- */

