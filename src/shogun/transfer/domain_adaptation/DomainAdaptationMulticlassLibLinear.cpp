/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Thoralf Klein, Heiko Strathmann, Soeren Sonnenburg
 */

#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/transfer/domain_adaptation/DomainAdaptationMulticlassLibLinear.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

DomainAdaptationMulticlassLibLinear::DomainAdaptationMulticlassLibLinear() :
	MulticlassLibLinear()
{
	init_defaults();
}

DomainAdaptationMulticlassLibLinear::DomainAdaptationMulticlassLibLinear(
		float64_t target_C, std::shared_ptr<DotFeatures> target_features, std::shared_ptr<Labels> target_labels,
		std::shared_ptr<LinearMulticlassMachine> source_machine) :
	MulticlassLibLinear(target_C,target_features,target_labels)
{
	init_defaults();

	set_source_machine(source_machine);
}

void DomainAdaptationMulticlassLibLinear::init_defaults()
{
	m_train_factor = 1.0;
	m_source_bias = 0.5;
	m_source_machine = NULL;

	register_parameters();
}

float64_t DomainAdaptationMulticlassLibLinear::get_source_bias() const
{
	return m_source_bias;
}

void DomainAdaptationMulticlassLibLinear::set_source_bias(float64_t source_bias)
{
	m_source_bias = source_bias;
}

float64_t DomainAdaptationMulticlassLibLinear::get_train_factor() const
{
	return m_train_factor;
}

void DomainAdaptationMulticlassLibLinear::set_train_factor(float64_t train_factor)
{
	m_train_factor = train_factor;
}

std::shared_ptr<LinearMulticlassMachine> DomainAdaptationMulticlassLibLinear::get_source_machine() const
{

	return m_source_machine;
}

void DomainAdaptationMulticlassLibLinear::set_source_machine(
		std::shared_ptr<LinearMulticlassMachine> source_machine)
{


	m_source_machine = source_machine;
}

void DomainAdaptationMulticlassLibLinear::register_parameters()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_source_machine, "source_machine", "source domain machine");
	SG_ADD(&m_train_factor, "train_factor", "factor of target domain regularization",
			ParameterProperties::HYPER);
	SG_ADD(&m_source_bias, "source_bias", "bias to source domain",
			ParameterProperties::HYPER);
}

DomainAdaptationMulticlassLibLinear::~DomainAdaptationMulticlassLibLinear()
{
}

SGMatrix<float64_t> DomainAdaptationMulticlassLibLinear::obtain_regularizer_matrix() const
{
	ASSERT(get_use_bias()==false)
	int32_t n_classes = m_source_machine->get_labels()->as<MulticlassLabels>()->get_num_classes();
	int32_t n_features = m_source_machine->get_features()->as<DotFeatures>()->get_dim_feature_space();
	SGMatrix<float64_t> w0(n_classes,n_features);

	for (int32_t i=0; i<n_classes; i++)
	{
		SGVector<float64_t> w = m_source_machine->get_machine(i)->as<LinearMachine>()->get_w();
		for (int32_t j=0; j<n_features; j++)
			w0(j,i) = m_train_factor*w[j];
	}

	return w0;
}

std::shared_ptr<BinaryLabels> DomainAdaptationMulticlassLibLinear::get_submachine_outputs(int32_t i)
{
	auto target_outputs = MulticlassMachine::get_submachine_outputs(i);
	auto source_outputs = m_source_machine->get_submachine_outputs(i);
	int32_t n_target_outputs = target_outputs->get_num_labels();
	ASSERT(n_target_outputs==source_outputs->get_num_labels())
	SGVector<float64_t> result(n_target_outputs);
	for (int32_t j=0; j<result.vlen; j++)
		result[j] = (1-m_source_bias)*target_outputs->get_value(j) + m_source_bias*source_outputs->get_value(j);

	return std::make_shared<BinaryLabels>(result);
}
#endif /* HAVE_LAPACK */
