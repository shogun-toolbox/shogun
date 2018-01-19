/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein
 */

#include <shogun/base/init.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

void test_sigmoid_fitting()
{
	CBinaryLabels* labels=new CBinaryLabels(10);
	labels->set_values(SGVector<float64_t>(labels->get_num_labels()));

	for (index_t i=0; i<labels->get_num_labels(); ++i)
		labels->set_value(i%2==0 ? 1 : -1, i);

	labels->get_values().display_vector("scores");
	labels->scores_to_probabilities();
	labels->get_values().display_vector("probabilities");


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

