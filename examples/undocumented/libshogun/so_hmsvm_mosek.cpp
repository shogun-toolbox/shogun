#include <shogun/labels/StructuredLabels.h>
#include <shogun/loss/HingeLoss.h>
#include <shogun/structure/HMSVMLabels.h>
#include <shogun/structure/HMSVMModel.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <shogun/structure/TwoStateModel.h>

using namespace shogun;

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();
#ifdef USE_MOSEK

	CHMSVMModel* model = CTwoStateModel::simulate_two_state_data();

	CStructuredLabels* labels = model->get_labels();
	CFeatures* features = model->get_features();

	CHingeLoss* loss = new CHingeLoss();

	CPrimalMosekSOSVM* sosvm = new CPrimalMosekSOSVM(model, loss, labels, features);
	SG_REF(sosvm);

	sosvm->train();
//	sosvm->get_w().display_vector("w");

	CStructuredLabels* out = CStructuredLabels::obtain_from_generic(sosvm->apply());

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

	SG_UNREF(l);

#endif /* USE_MOSEK */
	exit_shogun();

	return 0;
}
