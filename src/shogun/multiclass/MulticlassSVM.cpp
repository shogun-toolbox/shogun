/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/multiclass/MulticlassSVM.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>

using namespace shogun;

MulticlassSVM::MulticlassSVM()
	:MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>())
{
}

MulticlassSVM::MulticlassSVM(std::shared_ptr<MulticlassStrategy >strategy)
	:KernelMulticlassMachine(strategy, NULL, std::make_shared<SVM>(0), NULL)
{
	init();
}

MulticlassSVM::MulticlassSVM(
	std::shared_ptr<MulticlassStrategy >strategy, float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
	: KernelMulticlassMachine(strategy, k, std::make_shared<SVM>(C, k, lab), lab)
{
	init();
	m_C=C;
}

MulticlassSVM::~MulticlassSVM()
{
}

void MulticlassSVM::init()
{
	SG_ADD(&m_C, "C", "C regularization constant",ParameterProperties::HYPER);
	m_C=0;
}

bool MulticlassSVM::create_multiclass_svm(int32_t num_classes)
{
	if (num_classes>0)
	{
		m_machines.clear();
		return true;
	}
	return false;
}

bool MulticlassSVM::set_svm(int32_t num, std::shared_ptr<SVM> svm)
{
	if (!m_machines.empty() && m_machines.size()>num && num>=0 && svm)
	{
		m_machines.insert(m_machines.begin()+num, svm);
		return true;
	}
	return false;
}

bool MulticlassSVM::init_machines_for_apply(std::shared_ptr<Features> data)
{
	if (is_data_locked())
	{
		SG_ERROR("KernelMachine::apply(Features*) cannot be called when "
				"data_lock was called before. Call data_unlock to allow.");
	}

	if (!m_kernel)
		SG_ERROR("No kernel assigned!\n")

	auto lhs=m_kernel->get_lhs();
	if (!lhs && m_kernel->get_kernel_type()!=K_COMBINED)
		SG_ERROR("%s: No left hand side specified\n", get_name())

	if (m_kernel->get_kernel_type()!=K_COMBINED && !lhs->get_num_vectors())
	{
		SG_ERROR("%s: No vectors on left hand side (%s). This is probably due to"
				" an implementation error in %s, where it was forgotten to set "
				"the data (m_svs) indices\n", get_name(),
				data->get_name());
	}

	if (data && m_kernel->get_kernel_type()!=K_COMBINED)
		m_kernel->init(lhs, data);

	for (auto m: m_machines)
	{
		auto the_svm = m->as<SVM>();
		ASSERT(the_svm)
		the_svm->set_kernel(m_kernel);
	}

	return true;
}

