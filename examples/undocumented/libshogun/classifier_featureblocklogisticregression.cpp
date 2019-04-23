#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/FeatureBlockLogisticRegression.h>
#include <shogun/lib/IndexBlock.h>
#include <shogun/lib/IndexBlockTree.h>
#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

int main(int argc, char** argv)
{
	// create some data
	SGMatrix<float64_t> matrix(4,4);
	for (int32_t i=0; i<4*4; i++)
		matrix.matrix[i]=i;

	DenseFeatures<float64_t>* features= new DenseFeatures<float64_t>(matrix);

	// create three labels
	BinaryLabels* labels=new BinaryLabels(4);
	labels->set_label(0, -1);
	labels->set_label(1, +1);
	labels->set_label(2, -1);
	labels->set_label(3, +1);

	CIndexBlock* first_block = new CIndexBlock(0,2);
	CIndexBlock* second_block = new CIndexBlock(2,4);
	CIndexBlockGroup* block_group = new CIndexBlockGroup();
	block_group->add_block(first_block);
	block_group->add_block(second_block);

	CFeatureBlockLogisticRegression* regressor = new CFeatureBlockLogisticRegression(0.5,features,labels,block_group);
	regressor->train();

	regressor->get_w().display_vector();

	CIndexBlock* root_block = new CIndexBlock(0,4);
	root_block->add_sub_block(first_block);
	root_block->add_sub_block(second_block);
	CIndexBlockTree* block_tree = new CIndexBlockTree(root_block);

	regressor->set_feature_relation(block_tree);
	regressor->train();

	regressor->get_w().display_vector();

	return 0;
}
#else //USE_GPL_SHOGUN
int main(int argc, char** argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
