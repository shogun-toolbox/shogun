#include <labels/RegressionLabels.h>
#include <features/DenseFeatures.h>
#include <transfer/multitask/MultitaskLeastSquaresRegression.h>
#include <transfer/multitask/Task.h>
#include <transfer/multitask/TaskTree.h>
#include <transfer/multitask/TaskGroup.h>
#include <base/init.h>
#include <lib/common.h>
#include <io/SGIO.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message);

	// create some data
	SGMatrix<float64_t> matrix(2,4);
	for (int32_t i=0; i<2*4; i++)
		matrix.matrix[i]=i;

	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>(matrix);

	// create three labels
	CRegressionLabels* labels=new CRegressionLabels(4);
	labels->set_label(0, -1.4);
	labels->set_label(1, +1.5);
	labels->set_label(2, -1.2);
	labels->set_label(3, +1.1);

	CTask* first_task = new CTask(0,2);
	CTask* second_task = new CTask(2,4);

	CTaskGroup* task_group = new CTaskGroup();
	task_group->append_task(first_task);
	task_group->append_task(second_task);

	CMultitaskLeastSquaresRegression* regressor = new CMultitaskLeastSquaresRegression(0.5,features,labels,task_group);
	regressor->train();

	regressor->set_current_task(0);
	regressor->get_w().display_vector();
	SG_UNREF(regressor);
	exit_shogun();
	return 0;
}
