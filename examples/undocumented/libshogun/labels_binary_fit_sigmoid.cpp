/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein
 */

#include <shogun/evaluation/SigmoidCalibration.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

void test_sigmoid_fitting()
{
	BinaryLabels* labels=new BinaryLabels(10);
	BinaryLabels* predictions = new BinaryLabels(10);
	labels->set_values(SGVector<float64_t>(labels->get_num_labels()));

	for (index_t i=0; i<labels->get_num_labels(); ++i)
	{
		predictions->set_value(i % 2 == 0 ? 1 : -1, i);
		labels->set_value(i % 4 == 0 ? 1 : -1, i);
	}
	predictions->get_values().display_vector("scores");
	SigmoidCalibration* sigmoid_calibration = new SigmoidCalibration();
	sigmoid_calibration->fit_binary(predictions, labels);
	BinaryLabels* calibrated_labels =
	    sigmoid_calibration->calibrate_binary(predictions);
	calibrated_labels->get_values().display_vector("probabilities");

}

int main()
{
//	env()->io()->set_loglevel(MSG_DEBUG);

	test_sigmoid_fitting();

	return 0;
}

