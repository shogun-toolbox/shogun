#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/transfer/multitask/MultitaskLeastSquaresRegression.h>
#include <shogun/transfer/multitask/Task.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

int main(int argc, char** argv)
{
	// create some data
	SGMatrix<float64_t> matrix(2,4);
	for (int32_t i=0; i<2*4; i++)
		matrix.matrix[i]=i;

	DenseFeatures<float64_t>* features= new DenseFeatures<float64_t>(matrix);

	// create three labels
	RegressionLabels* labels=new RegressionLabels(4);
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
	return 0;
}
#else //USE_GPL_SHOGUN
int main(int argc, char** argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
