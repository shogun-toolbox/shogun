#include <shogun/features/MatrixFeatures.h>
#include <shogun/loss/HingeLoss.h>
#include <shogun/structure/SequenceLabels.h>
#include <shogun/structure/HMSVMModel.h>
#include <shogun/structure/PrimalMosekSOSVM.h>

using namespace shogun;


int main(int argc, char ** argv)
{
	init_shogun_with_defaults();
#ifdef USE_MOSEK

	// Create structured labels
	CSequenceLabels* labels = new CSequenceLabels(5, 2);

	// Label sequences of with two states
	int32_t lab1[] = {0, 0, 1, 1};
	int32_t lab2[] = {1, 1, 1, 0};
	int32_t lab3[] = {0, 1, 0, 1};
	int32_t lab4[] = {1, 0, 0, 0};
	int32_t lab5[] = {0, 1, 1, 0};

	// No need for ref_counting in SGVector since the data is allocated
	// during compilation time
	labels->add_vector_label(SGVector< int32_t >(lab1, 4, false));
	labels->add_vector_label(SGVector< int32_t >(lab2, 4, false));
	labels->add_vector_label(SGVector< int32_t >(lab3, 4, false));
	labels->add_vector_label(SGVector< int32_t >(lab4, 4, false));
	labels->add_vector_label(SGVector< int32_t >(lab5, 4, false));

	// Create features
	CMatrixFeatures< float64_t >* features = new CMatrixFeatures< float64_t >(5, 3);

	// Observation matrices with three states
	float64_t mat1[] = { 0., 1., 2., 1., 1., 1., 2., 2., 2., 1., 0., 1. };
	float64_t mat2[] = { 1., 2., 2., 0., 2., 1., 1., 1., 0., 0., 2., 1. };
	float64_t mat3[] = { 0., 1., 2., 1., 1., 2., 1., 1., 0., 0., 1., 0. };
	float64_t mat4[] = { 1., 2., 1., 0., 2., 1., 0., 2., 0., 1., 0., 2. };
	float64_t mat5[] = { 2., 2., 0., 1., 2., 1., 0., 1., 2., 0., 2., 0. };

	features->set_feature_vector(SGMatrix< float64_t >(mat1, 3, 4, false), 0);
	features->set_feature_vector(SGMatrix< float64_t >(mat2, 3, 4, false), 1);
	features->set_feature_vector(SGMatrix< float64_t >(mat3, 3, 4, false), 2);
	features->set_feature_vector(SGMatrix< float64_t >(mat4, 3, 4, false), 3);
	features->set_feature_vector(SGMatrix< float64_t >(mat5, 3, 4, false), 4);

	CHMSVMModel* model = new CHMSVMModel(features, labels, SMT_TWO_STATE, 3);
	SG_REF(model);
	CPrimalMosekSOSVM* sosvm = new CPrimalMosekSOSVM(model, labels);
	SG_REF(sosvm);

	sosvm->train();

	sosvm->get_w().display_vector("w");
	sosvm->get_slacks().display_vector("slacks");

	// Free memory
	SG_UNREF(sosvm);
	SG_UNREF(model);

#endif /* USE_MOSEK */
	exit_shogun();

	return 0;
}

