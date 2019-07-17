#include <shogun/structure/TwoStateModel.h>
#include <shogun/structure/HMSVMModel.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>

using namespace shogun;

int main()
{
	CTwoStateModel* tsm = new CTwoStateModel();
	CHMSVMModel* model = tsm->simulate_data(100,250,3,1);
	CStructuredLabels* labels = model->get_labels();
	CDualLibQPBMSOSVM* sosvm = new CDualLibQPBMSOSVM(model, labels, 5000.0);
	sosvm->train();

	SG_UNREF(sosvm);
	SG_UNREF(labels);
	SG_UNREF(tsm);
	return 0;
}
