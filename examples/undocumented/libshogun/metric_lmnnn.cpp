#include <shogun/metric/LMNN.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	// create features, each column is a feature vector
	SGMatrix<float64_t> feat_mat(2,4);
	// 1st feature vector
	feat_mat(0,0)=0;
	feat_mat(1,0)=0;
	// 2nd feature vector
	feat_mat(0,1)=0;
	feat_mat(1,1)=-1;
	// 3rd feature vector
	feat_mat(0,2)=1;
	feat_mat(1,2)=1;
	// 4th feature vector
	feat_mat(0,3)=-1;
	feat_mat(1,3)=1;
	// wrap feat_mat into Shogun features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat_mat);

	// create labels
	SGVector<float64_t> lab_vec(4);
	lab_vec[0]=0;
	lab_vec[1]=0;
	lab_vec[2]=1;
	lab_vec[3]=1;
	// two-class data, use MulticlassLabels because LMNN works in general for more than two classes
	CMulticlassLabels* labels=new CMulticlassLabels(lab_vec);

	// create LMNN metric machine
	int32_t k=1;	// number of target neighbors per example
	CLMNN* lmnn=new CLMNN(features,labels,k);
	// use the identity matrix as initial transform for LMNN
	SGMatrix<float64_t> init_transform=SGMatrix<float64_t>::create_identity_matrix(2,1);
	// set number of maximum iterations and train
	lmnn->set_maxiter(1500);
//	lmnn->io->set_loglevel(MSG_DEBUG);
	lmnn->train(init_transform);
//	lmnn->get_linear_transform().display_matrix("linear_transform");
	CLMNNStatistics* statistics=lmnn->get_statistics();
/*
	statistics->obj.display_vector("objective");
	statistics->stepsize.display_vector("stepsize");
	statistics->num_impostors.display_vector("num_impostors");
*/

	SG_UNREF(statistics);
	SG_UNREF(lmnn);
	exit_shogun();
}
