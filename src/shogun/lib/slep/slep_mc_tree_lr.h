/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (W) 2012 Sergey Lisitsyn
 */

#ifndef SLEP_MC_TREE_LR_H_
#define SLEP_MC_TREE_LR_H_
#include <shogun/lib/config.h>
#include <shogun/lib/slep/slep_options.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/MulticlassLabels.h>

namespace shogun
{

/** Accelerated projected gradient solver for multiclass
 * logistic regression problem with feature tree regularization.
 *
 * @param features features to be used
 * @param labels labels to be used
 * @param z regularization ratio
 * @param options options of solver
 */
slep_result_t slep_mc_tree_lr(
		CDotFeatures* features,
		CMulticlassLabels* labels,
		float64_t z,
		const slep_options& options);

};
#endif /* SLEP_MC_TREE_LR_H_ */
