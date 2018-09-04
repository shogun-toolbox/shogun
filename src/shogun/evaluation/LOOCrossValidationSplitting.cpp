/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre
 */

#include <shogun/evaluation/LOOCrossValidationSplitting.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

CLOOCrossValidationSplitting::CLOOCrossValidationSplitting() :
	CCrossValidationSplitting()
{}

CLOOCrossValidationSplitting::CLOOCrossValidationSplitting(
		CLabels* labels) :
	CCrossValidationSplitting(labels, labels->get_num_labels())
{}
