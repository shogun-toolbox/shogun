#include <shogun/structure/TwoStateModel.h>
#include <shogun/structure/HMSVMModel.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>

using namespace shogun;

int main()
{
	TwoStateModel* tsm = new TwoStateModel();
	CHMSVMModel* model = tsm->simulate_data(100,250,3,1);
	StructuredLabels* labels = model->get_labels();
	CDualLibQPBMSOSVM* sosvm = new CDualLibQPBMSOSVM(model, labels, 5000.0);
	sosvm->train();

	return 0;
}
