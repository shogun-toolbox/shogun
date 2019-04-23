/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK

#include <shogun/transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <iostream>
#include <vector>

using namespace shogun;


DomainAdaptationSVMLinear::DomainAdaptationSVMLinear() : LibLinear(L2R_L1LOSS_SVC_DUAL)
{
	init(NULL, 0.0);
}


DomainAdaptationSVMLinear::DomainAdaptationSVMLinear(float64_t C, std::shared_ptr<DotFeatures> f, std::shared_ptr<Labels> lab, std::shared_ptr<LinearMachine> pre_svm, float64_t B_param) : LibLinear(C, f, lab)
{
	init(pre_svm, B_param);

}


DomainAdaptationSVMLinear::~DomainAdaptationSVMLinear()
{


	SG_DEBUG("deleting DomainAdaptationSVMLinear\n")
}


void DomainAdaptationSVMLinear::init(std::shared_ptr<LinearMachine> pre_svm, float64_t B_param)
{

	if (pre_svm)
	{
		// increase reference counts


		// set bias of parent svm to zero
		pre_svm->set_bias(0.0);
	}

	this->presvm = pre_svm;
	this->B = B_param;
	this->train_factor = 1.0;

	set_liblinear_solver_type(L2R_L1LOSS_SVC_DUAL);

	// invoke sanity check
	is_presvm_sane();

    // serialization code
	SG_ADD(&presvm, "presvm", "SVM to regularize against", ParameterProperties::HYPER);
	SG_ADD(&B, "B", "Regularization strenth B.", ParameterProperties::HYPER);
	SG_ADD(&train_factor, "train_factor", "train_factor", ParameterProperties::HYPER);
}


bool DomainAdaptationSVMLinear::is_presvm_sane()
{

	if (!presvm) {

		SG_WARNING("presvm is null")

	} else {

        if (presvm->get_bias() != 0) {
            SG_ERROR("presvm bias not set to zero")
        }

        if (presvm->get_features()->get_feature_type() != this->get_features()->get_feature_type()) {
            SG_ERROR("feature types do not agree")
        }
    }

	return true;

}


bool DomainAdaptationSVMLinear::train_machine(std::shared_ptr<Features> train_data)
{

	std::shared_ptr<DotFeatures> tmp_data;

	if (m_labels->get_label_type() != LT_BINARY)
		SG_ERROR("DomainAdaptationSVMLinear requires binary labels\n")

	if (train_data)
	{
		if (!train_data->has_property(FP_DOT))
			SG_ERROR("DotFeatures expected\n")

		if (m_labels->as<BinaryLabels>()->get_num_labels() != train_data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")

		tmp_data = train_data->as<DotFeatures>();
	}
	else
	{
		tmp_data = features;
	}

	auto labels = binary_labels(get_labels());
	int32_t num_training_points = labels->get_num_labels();

	std::vector<float64_t> lin_term = std::vector<float64_t>(num_training_points);

    if (presvm)
    {
	ASSERT(presvm->get_bias() == 0.0)

        // bias of parent SVM was set to zero in constructor, already contains B
        auto parent_svm_out = presvm->apply_binary(tmp_data);

        SG_DEBUG("pre-computing linear term from presvm\n")

        // pre-compute linear term
        for (int32_t i=0; i!=num_training_points; i++)
        {
            lin_term[i] = train_factor * B * labels->get_value(i) * parent_svm_out->get_value(i) - 1.0;
        }

	// set linear term for QP
		this->set_linear_term(
				SGVector<float64_t>(&lin_term[0], lin_term.size()));

    }



	/*
	// warm-start liblinear
	//TODO test this code, measure speed-ups
    //presvm w stored in presvm
    float64_t* tmp_w;
    presvm->get_w(tmp_w, w_dim);

    //copy vector
    float64_t* tmp_w_copy = SG_MALLOC(float64_t, w_dim);
    std::copy(tmp_w, tmp_w + w_dim, tmp_w_copy);

	for (int32_t i=0; i!=w_dim; i++)
	{
		tmp_w_copy[i] = B * tmp_w_copy[i];
	}

	//set w (copied in setter)
    set_w(tmp_w_copy, w_dim);
    SG_FREE(tmp_w_copy);
	*/

	bool success = false;

	//train SVM
	if (train_data)
	{
		success = LibLinear::train_machine(train_data);
	} else {
		success = LibLinear::train_machine();
	}

	//ASSERT(presvm)

	return success;

}


std::shared_ptr<LinearMachine> DomainAdaptationSVMLinear::get_presvm()
{
	return presvm;
}


float64_t DomainAdaptationSVMLinear::get_B()
{
	return B;
}


float64_t DomainAdaptationSVMLinear::get_train_factor()
{
	return train_factor;
}


void DomainAdaptationSVMLinear::set_train_factor(float64_t factor)
{
	train_factor = factor;
}


std::shared_ptr<BinaryLabels> DomainAdaptationSVMLinear::apply_binary(std::shared_ptr<Features> data)
{
	ASSERT(presvm->get_bias()==0.0)

	int32_t num_examples = data->get_num_vectors();

	auto out_current = LibLinear::apply_binary(data);

	SGVector<float64_t> out_combined(num_examples);
	if (presvm)
	{
		// recursive call if used on DomainAdaptationSVM object
		auto out_presvm = presvm->apply_binary(data);


		// combine outputs
		for (int32_t i=0; i!=num_examples; i++)
			out_combined[i] = out_current->get_value(i) + B*out_presvm->get_value(i);


	}



	return std::make_shared<BinaryLabels>(out_combined);
}

#endif //HAVE_LAPACK

