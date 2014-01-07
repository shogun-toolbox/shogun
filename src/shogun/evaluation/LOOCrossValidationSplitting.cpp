/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Saurabh Mahindre
 */

#include <evaluation/LOOCrossValidationSplitting.h>
#include <labels/Labels.h>

using namespace shogun;

CLOOCrossValidationSplitting::CLOOCrossValidationSplitting() :
	CCrossValidationSplitting()
{}

CLOOCrossValidationSplitting::CLOOCrossValidationSplitting(
		CLabels* labels) :
	CCrossValidationSplitting(labels, labels->get_num_labels())
{}
