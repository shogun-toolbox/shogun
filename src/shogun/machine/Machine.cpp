/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Giovanni De Toni,
 *          Soeren Sonnenburg, Chiyuan Zhang, Thoralf Klein, Evgeniy Andreev,
 *          Evan Shelhamer, Fernando Iglesias
 */

#include <rxcpp/rx-lite.hpp>
#include <shogun/base/init.h>
#include <shogun/lib/Signal.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

Machine::Machine()
    : StoppableSGObject(), m_max_train_time(0), m_labels(NULL),
      m_solver_type(ST_AUTO)
{
	m_data_locked = false;
	m_store_model_features = false;

	SG_ADD(&m_max_train_time, "max_train_time", "Maximum training time.");
	SG_ADD(&m_labels, "labels", "Labels to be used.");
	SG_ADD(
	    &m_store_model_features, "store_model_features",
	    "Should feature data of model be stored after training?");
	SG_ADD(&m_data_locked, "data_locked", "Indicates whether data is locked");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_solver_type, "solver_type", "Type of solver.",
	    ParameterProperties::NONE,
	    SG_OPTIONS(
	        ST_AUTO, ST_CPLEX, ST_GLPK, ST_NEWTON, ST_DIRECT, ST_ELASTICNET,
	        ST_BLOCK_NORM));
}

Machine::~Machine()
{

}

bool Machine::train(std::shared_ptr<Features> data)
{
	/* not allowed to train on locked data */
	REQUIRE(
	    !m_data_locked, "(%s)::train data_lock() was called, only "
	                    "train_locked() is possible. Call data_unlock if you "
	                    "want to call train()\n",
	    get_name());

	if (train_require_labels())
	{
		if (m_labels == NULL)
		{
			SG_ERROR("%s@%p: No labels given", get_name(), this)
		}
		m_labels->ensure_valid(get_name());
	}

	auto sub = connect_to_signal_handler();
	bool result = false;

	if (support_feature_dispatching())
	{
		REQUIRE(data != NULL, "Features not provided!");
		REQUIRE(
		    data->get_num_vectors() == m_labels->get_num_labels(),
		    "Number of training vectors (%d) does not match number of "
		    "labels (%d)\n",
		    data->get_num_vectors(), m_labels->get_num_labels())

		if (support_dense_dispatching() && data->get_feature_class() == C_DENSE)
			result = train_dense(data);
		else if (
		    support_string_dispatching() &&
		    data->get_feature_class() == C_STRING)
			result = train_string(data);
		else
			SG_ERROR("Training with %s is not implemented!", data->get_name());
	}
	else
		result = train_machine(data);

	sub.unsubscribe();
	reset_computation_variables();

	if (m_store_model_features)
		store_model_features();

	return result;
}

bool Machine::train_locked()
{
	/*train machine without any actual features(data is locked)*/
	REQUIRE(
	    is_data_locked(),
	    "Data needs to be locked for training, call data_lock()\n")

	auto sub = connect_to_signal_handler();
	bool result = train_machine();
	sub.unsubscribe();
	reset_computation_variables();
	return result;
}

void Machine::set_labels(std::shared_ptr<Labels> lab)
{
    if (lab != NULL)
    {
        if (!is_label_valid(lab))
            SG_ERROR("Invalid label for %s", get_name())
    }
    m_labels = lab;
}

std::shared_ptr<Labels> Machine::get_labels()
{
	return m_labels;
}

void Machine::set_max_train_time(float64_t t)
{
	m_max_train_time = t;
}

float64_t Machine::get_max_train_time()
{
	return m_max_train_time;
}

EMachineType Machine::get_classifier_type()
{
	return CT_NONE;
}

void Machine::set_solver_type(ESolverType st)
{
	m_solver_type = st;
}

ESolverType Machine::get_solver_type()
{
	return m_solver_type;
}

void Machine::set_store_model_features(bool store_model)
{
	m_store_model_features = store_model;
}

void Machine::data_lock(std::shared_ptr<Labels> labs, std::shared_ptr<Features> features)
{
	SG_DEBUG("entering %s::data_lock\n", get_name())
	if (!supports_locking())
	{
		{
			SG_ERROR("%s::data_lock(): Machine does not support data locking!\n",
					get_name());
		}
	}

	if (!labs)
	{
		SG_ERROR("%s::data_lock() is not possible will NULL labels!\n",
				get_name());
	}

	/* first set labels */
	set_labels(labs);

	if (m_data_locked)
	{
		SG_ERROR("%s::data_lock() was already called. Dont lock twice!",
				get_name());
	}

	m_data_locked=true;
	post_lock(labs,features);
	SG_DEBUG("leaving %s::data_lock\n", get_name())
}

void Machine::data_unlock()
{
	SG_DEBUG("entering %s::data_lock\n", get_name())
	if (m_data_locked)
		m_data_locked=false;

	SG_DEBUG("leaving %s::data_lock\n", get_name())
}

std::shared_ptr<Labels> Machine::apply(std::shared_ptr<Features> data)
{
	SG_DEBUG("entering %s::apply(%s at %p)\n",
			get_name(), data ? data->get_name() : "NULL", data.get());

	std::shared_ptr<Labels> result=NULL;

	switch (get_machine_problem_type())
	{
		case PT_BINARY:
			result=apply_binary(data);
			break;
		case PT_REGRESSION:
			result=apply_regression(data);
			break;
		case PT_MULTICLASS:
			result=apply_multiclass(data);
			break;
		case PT_STRUCTURED:
			result=apply_structured(data);
			break;
		case PT_LATENT:
			result=apply_latent(data);
			break;
		default:
			SG_ERROR("Unknown problem type")
			break;
	}

	SG_DEBUG("leaving %s::apply(%s at %p)\n",
			get_name(), data ? data->get_name() : "NULL", data.get());

	return result;
}

std::shared_ptr<Labels> Machine::apply_locked(SGVector<index_t> indices)
{
	switch (get_machine_problem_type())
	{
		case PT_BINARY:
			return apply_locked_binary(indices);
		case PT_REGRESSION:
			return apply_locked_regression(indices);
		case PT_MULTICLASS:
			return apply_locked_multiclass(indices);
		case PT_STRUCTURED:
			return apply_locked_structured(indices);
		case PT_LATENT:
			return apply_locked_latent(indices);
		default:
			SG_ERROR("Unknown problem type")
			break;
	}
	return NULL;
}

std::shared_ptr<BinaryLabels> Machine::apply_binary(std::shared_ptr<Features> data)
{
	SG_ERROR("This machine does not support apply_binary()\n")
	return NULL;
}

std::shared_ptr<RegressionLabels> Machine::apply_regression(std::shared_ptr<Features> data)
{
	SG_ERROR("This machine does not support apply_regression()\n")
	return NULL;
}

std::shared_ptr<MulticlassLabels> Machine::apply_multiclass(std::shared_ptr<Features> data)
{
	SG_ERROR("This machine does not support apply_multiclass()\n")
	return NULL;
}

std::shared_ptr<StructuredLabels> Machine::apply_structured(std::shared_ptr<Features> data)
{
	SG_ERROR("This machine does not support apply_structured()\n")
	return NULL;
}

std::shared_ptr<LatentLabels> Machine::apply_latent(std::shared_ptr<Features> data)
{
	SG_ERROR("This machine does not support apply_latent()\n")
	return NULL;
}

std::shared_ptr<BinaryLabels> Machine::apply_locked_binary(SGVector<index_t> indices)
{
	SG_ERROR("apply_locked_binary(SGVector<index_t>) is not yet implemented "
			"for %s\n", get_name());
	return NULL;
}

std::shared_ptr<RegressionLabels> Machine::apply_locked_regression(SGVector<index_t> indices)
{
	SG_ERROR("apply_locked_regression(SGVector<index_t>) is not yet implemented "
			"for %s\n", get_name());
	return NULL;
}

std::shared_ptr<MulticlassLabels> Machine::apply_locked_multiclass(SGVector<index_t> indices)
{
	SG_ERROR("apply_locked_multiclass(SGVector<index_t>) is not yet implemented "
			"for %s\n", get_name());
	return NULL;
}

std::shared_ptr<StructuredLabels> Machine::apply_locked_structured(SGVector<index_t> indices)
{
	SG_ERROR("apply_locked_structured(SGVector<index_t>) is not yet implemented "
			"for %s\n", get_name());
	return NULL;
}

std::shared_ptr<LatentLabels> Machine::apply_locked_latent(SGVector<index_t> indices)
{
	SG_ERROR("apply_locked_latent(SGVector<index_t>) is not yet implemented "
			"for %s\n", get_name());
	return NULL;
}
