/*
* Written (W) 2019 Giovanni De Toni
*/

#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/lib/parameter_observers/ParameterObserverLogger.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>

using namespace shogun;

// Data generator
void generate_data_n_greater_d(SGMatrix<float64_t> &data, SGVector<float64_t> &lab)
{
	data(0,0)=0.044550005575722;
	data(1,0)=-0.433969606728583;
	data(2,0)=-0.397935396933392;
	data(0,1)=-0.778754072066602;
	data(1,1)=-0.620105076569903;
	data(2,1)=-0.542538248707627;
	data(0,2)=0.334313094513960;
	data(1,2)=0.421985645755003;
	data(2,2)=0.263031426076997;
	data(0,3)=0.516043376162584;
	data(1,3)=0.159041471773470;
	data(2,3)=0.691318725364356;
	data(0,4)=-0.116152404185664;
	data(1,4)=0.473047565770014;
	data(2,4)=-0.013876505800334;

	lab[0]=-0.196155100498902;
	lab[1]=-5.376485285422094;
	lab[2]=-1.717489351713958;
	lab[3]=4.506538567065213;
	lab[4]=2.783591170569741;
}

int main() {

	init_shogun_with_defaults();

	SGMatrix<float64_t> data(3,5);
	SGVector<float64_t> lab(5);
	generate_data_n_greater_d(data, lab);

	// Set the parameter observer
	Some<ParameterObserverLogger> logger = some<ParameterObserverLogger>();

	// Set features and labels
	Some<CDenseFeatures<float64_t>> features= some<CDenseFeatures<float64_t>>(data);
	Some<CRegressionLabels> labels= some<CRegressionLabels>(lab);

	// Create LARS object
	Some<CLeastAngleRegression> lars= some<CLeastAngleRegression>();

	// Subscribe the observer to the LARS object
	lars->subscribe_to_parameters(logger);

	// Set labels and train
	lars->set_labels((CLabels*) labels);
	lars->train(features);

	SGVector<float64_t> active3=SGVector<float64_t>(lars->get_w_for_var(3));
	SGVector<float64_t> active2=SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1=SGVector<float64_t>(lars->get_w_for_var(1));

	return 0;
}
