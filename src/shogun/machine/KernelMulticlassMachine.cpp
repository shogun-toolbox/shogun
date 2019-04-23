/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Heiko Strathmann, Sergey Lisitsyn, Thoralf Klein,
 *          Soeren Sonnenburg
 */

#include <shogun/lib/Set.h>
#include <shogun/machine/KernelMulticlassMachine.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/KernelMachine.h>

using namespace shogun;

void KernelMulticlassMachine::store_model_features()
{
	auto kernel= m_kernel;
	if (!kernel)
		SG_ERROR("%s::store_model_features(): kernel is needed to store SV "
				"features.\n", get_name());

	auto lhs=kernel->get_lhs();
	auto rhs=kernel->get_rhs();
	if (!lhs)
	{
		SG_ERROR("%s::store_model_features(): kernel lhs is needed to store "
		"SV features.\n", get_name());
	}

	/* this map will be abused as a map */
	Set<index_t> all_sv;
	for (auto m: m_machines)
	{
		auto machine=m->as<KernelMachine>();
		for (index_t j=0; j<machine->get_num_support_vectors(); ++j)
			all_sv.add(machine->get_support_vector(j));


	}

	/* convert map to vector of SV */
	SGVector<index_t> sv_idx(all_sv.get_num_elements());
	for (index_t i=0; i<sv_idx.vlen; ++i)
		sv_idx[i]=*all_sv.get_element_ptr(i);

	auto sv_features=lhs->copy_subset(sv_idx);

	/* now, features are replaced by concatenated SV features */
	kernel->init(sv_features, rhs);

	/* was SG_REF'ed by copy_subset */


	/* now the old SV indices have to be mapped to the new features */

	/* update SV of all machines */
	for (auto m: m_machines)
	{
		auto machine=m->as<KernelMachine>();

		/* for each machine, replace SV by index in sv_idx array */
		for (int32_t j=0; j<machine->get_num_support_vectors(); ++j)
		{
			/* get index of SV in old features */
			index_t current_sv_idx=machine->get_support_vector(j);

			/* the position of this old index in the map is the position of
			 * the SV in the new features */
			index_t new_sv_idx=all_sv.index_of(current_sv_idx);

			machine->set_support_vector(j, new_sv_idx);
		}


	}



}

KernelMulticlassMachine::KernelMulticlassMachine() : MulticlassMachine(), m_kernel(NULL)
{
	SG_ADD(&m_kernel,"kernel", "The kernel to be used", ParameterProperties::HYPER);
}

/** standard constructor
 * @param strategy multiclass strategy
 * @param kernel kernel
 * @param machine kernel machine
 * @param labs labels
 */
KernelMulticlassMachine::KernelMulticlassMachine(std::shared_ptr<MulticlassStrategy >strategy, std::shared_ptr<Kernel> kernel, std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labs) :
	MulticlassMachine(strategy,machine,labs), m_kernel(NULL)
{
	set_kernel(kernel);
	SG_ADD(&m_kernel,"kernel", "The kernel to be used", ParameterProperties::HYPER);
}

/** destructor */
KernelMulticlassMachine::~KernelMulticlassMachine()
{

}

/** set kernel
 *
 * @param k kernel
 */
void KernelMulticlassMachine::set_kernel(std::shared_ptr<Kernel> k)
{
	m_machine->as<KernelMachine>()->set_kernel(k);


	m_kernel=k;
}

std::shared_ptr<Kernel> KernelMulticlassMachine::get_kernel() const
{

	return m_kernel;
}

bool KernelMulticlassMachine::init_machine_for_train(std::shared_ptr<Features> data)
{
	if (data)
		m_kernel->init(data,data);

	m_machine->as<KernelMachine>()->set_kernel(m_kernel);

	return true;
}

bool KernelMulticlassMachine::init_machines_for_apply(std::shared_ptr<Features> data)
{
	if (data)
	{
		/* set data to rhs for this kernel */
		auto lhs=m_kernel->get_lhs();
		m_kernel->init(lhs, data);

	}

	/* set kernel to all sub-machines */
	for (auto m: m_machines)
	{
		auto machine=m->as<KernelMachine>();
		machine->set_kernel(m_kernel);
	}

	return true;
}

bool KernelMulticlassMachine::is_ready()
{
	if (m_kernel && m_kernel->get_num_vec_lhs() && m_kernel->get_num_vec_rhs())
			return true;

	return false;
}

std::shared_ptr<Machine> KernelMulticlassMachine::get_machine_from_trained(std::shared_ptr<Machine> machine) const
{
	return std::make_shared<KernelMachine>(machine->as<KernelMachine>());
}

int32_t KernelMulticlassMachine::get_num_rhs_vectors() const
{
	return m_kernel->get_num_vec_rhs();
}

void KernelMulticlassMachine::add_machine_subset(SGVector<index_t> subset)
{
	SG_NOTIMPLEMENTED
}

void KernelMulticlassMachine::remove_machine_subset()
{
	SG_NOTIMPLEMENTED
}


