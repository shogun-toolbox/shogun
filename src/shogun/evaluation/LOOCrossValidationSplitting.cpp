/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre
 */

#include <shogun/evaluation/LOOCrossValidationSplitting.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

LOOCrossValidationSplitting::LOOCrossValidationSplitting() :
	CrossValidationSplitting()
{}

LOOCrossValidationSplitting::LOOCrossValidationSplitting(
		std::shared_ptr<Labels> labels) :
	CrossValidationSplitting(labels, labels->get_num_labels())
{}
