#include <structure/HMSVMModel.h>
#include <structure/DualLibQPBMSOSVM.h>
#include <structure/StateModelTypes.h>
#include <features/MatrixFeatures.h>


using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	float64_t features_dat[] = {0,1,1, 2,1,2, 0,1,0, 0,2,2};
	SGMatrix<float64_t> features_mat(features_dat,1,12,false);
	CMatrixFeatures<float64_t>* features = new CMatrixFeatures<float64_t>(features_mat,3,4);

	int32_t labels_dat[] = {0,0,0, 1,1,1, 0,0,0, 1,1,1};
	SGVector<int32_t> labels_vec(labels_dat,12,false);
	CSequenceLabels* labels = new CSequenceLabels(labels_vec,3,4,2);
	labels->io->set_loglevel(MSG_DEBUG);

	CHMSVMModel* model = new CHMSVMModel(features, labels, SMT_TWO_STATE, 3);
	CDualLibQPBMSOSVM* sosvm = new CDualLibQPBMSOSVM(model, labels, 5000,0);
	sosvm->train();

	SG_UNREF(sosvm);
	exit_shogun();
	return 0;
}
