#include <shogun/structure/HMSVMModel.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/StateModelTypes.h>
#include <shogun/features/MatrixFeatures.h>


using namespace shogun;

int main()
{
	init_shogun_with_defaults();
	
	float64_t features_dat[] = {0,1, 2,1, 0,1, 0,2};
	SGMatrix<float64_t> features_mat(features_dat,1,8,false);
	CMatrixFeatures<float64_t>* features = new CMatrixFeatures<float64_t>(features_mat,2,4);

	int32_t labels_dat[] = {0,0, 1,1, 0,0, 1,1};
	SGVector<int32_t> labels_vec(labels_dat,8,false);
	CSequenceLabels* labels = new CSequenceLabels(labels_vec,2,4,2);

	CHMSVMModel* model = new CHMSVMModel(features, labels, SMT_TWO_STATE, 3);
	CDualLibQPBMSOSVM* sosvm = new CDualLibQPBMSOSVM(model, labels, 5000,0);
	sosvm->train();

	SG_UNREF(sosvm);
	exit_shogun();
	return 0;
}
