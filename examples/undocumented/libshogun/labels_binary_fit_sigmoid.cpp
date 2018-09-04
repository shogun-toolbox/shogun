/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/SigmoidCalibration.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

void test_sigmoid_fitting()
{
	CBinaryLabels* labels=new CBinaryLabels(10);
	CBinaryLabels* predictions = new CBinaryLabels(10);
	labels->set_values(SGVector<float64_t>(labels->get_num_labels()));

	for (index_t i=0; i<labels->get_num_labels(); ++i)
	{
		predictions->set_value(i % 2 == 0 ? 1 : -1, i);
		labels->set_value(i % 4 == 0 ? 1 : -1, i);
	}
	predictions->get_values().display_vector("scores");
	CSigmoidCalibration* sigmoid_calibration = new CSigmoidCalibration();
	sigmoid_calibration->fit_binary(predictions, labels);
	CBinaryLabels* calibrated_labels =
	    sigmoid_calibration->calibrate_binary(predictions);
	calibrated_labels->get_values().display_vector("probabilities");

	SG_UNREF(predictions);
	SG_UNREF(sigmoid_calibration);
	SG_UNREF(calibrated_labels);
	SG_UNREF(labels);
}

int main()
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	test_sigmoid_fitting();

	exit_shogun();
	return 0;
}

