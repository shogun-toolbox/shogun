#include <shogun/labels/StructuredLabels.h>
#include <shogun/labels/LabelsFactory.h>
#include <shogun/structure/HMSVMModel.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <shogun/structure/TwoStateModel.h>

using namespace shogun;

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();
#ifdef USE_MOSEK

	int32_t num_examples = 10;
	int32_t example_length = 250;
	int32_t num_features = 10;
	int32_t num_noise_features = 2;
	CHMSVMModel* model = CTwoStateModel::simulate_data(num_examples, example_length, num_features, num_noise_features);

	CStructuredLabels* labels = model->get_labels();
	CFeatures* features = model->get_features();

	CPrimalMosekSOSVM* sosvm = new CPrimalMosekSOSVM(model, labels);
	SG_REF(sosvm);

	sosvm->train();
//	sosvm->get_w().display_vector("w");

	CStructuredLabels* out = CLabelsFactory::to_structured(sosvm->apply());

	ASSERT( out->get_num_labels() == labels->get_num_labels() );

	for ( int32_t i = 0 ; i < out->get_num_labels() ; ++i )
	{
		CSequence* pred_seq = CSequence::obtain_from_generic( out->get_label(i) );
		CSequence* true_seq = CSequence::obtain_from_generic( labels->get_label(i) );
		SG_UNREF(pred_seq);
		SG_UNREF(true_seq);
	}

	SG_UNREF(out);
	SG_UNREF(features); // because model->get_features() increased the count
	SG_UNREF(labels);   // because model->get_labels() increased the count
	SG_UNREF(sosvm);

#endif /* USE_MOSEK */
	exit_shogun();

	return 0;
}
