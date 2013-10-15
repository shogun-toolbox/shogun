#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>
#include <shogun/transfer/multitask/Task.h>
#include <shogun/transfer/multitask/TaskTree.h>
#include <shogun/transfer/multitask/TaskGroup.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	// create some data
	SGMatrix<float64_t> matrix(2,4);
	for (int32_t i=0; i<2*4; i++)
		matrix.matrix[i]=i;

	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>(matrix);

	// create three labels
	CBinaryLabels* labels=new CBinaryLabels(4);
	labels->set_label(0, -1);
	labels->set_label(1, +1);
	labels->set_label(2, -1);
	labels->set_label(3, +1);

	CTask* first_task = new CTask(0,2);
	CTask* second_task = new CTask(2,4);
	CTaskGroup* task_group = new CTaskGroup();
	task_group->append_task(first_task);
	task_group->append_task(second_task);

	CMultitaskLogisticRegression* regressor = new CMultitaskLogisticRegression(0.5,features,labels,task_group);
	regressor->train();

	regressor->set_current_task(0);
	regressor->get_w().display_vector();

	CTask* root_task = new CTask(0,4);
	root_task->add_subtask(first_task);
	root_task->add_subtask(second_task);
	CTaskTree* task_tree = new CTaskTree(root_task);

	regressor->set_task_relation(task_tree);
	regressor->train();

	regressor->set_current_task(0);
	regressor->get_w().display_vector();

	SG_UNREF(regressor);
	exit_shogun();
	return 0;
}
