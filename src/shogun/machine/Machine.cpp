/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Giovanni De Toni, 
 *          Soeren Sonnenburg, Chiyuan Zhang, Thoralf Klein, Evgeniy Andreev, 
 *          Evan Shelhamer, Fernando Iglesias
 */

#include <rxcpp/rx-lite.hpp>
#include <shogun/lib/Signal.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

CMachine::CMachine()
    : CStoppableSGObject(), m_max_train_time(0), m_labels(NULL),
      m_solver_type(ST_AUTO)
{
	SG_ADD(&m_max_train_time, "max_train_time", "Maximum training time.");
	SG_ADD(&m_labels, "labels", "Labels to be used.");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_solver_type, "solver_type", "Type of solver.",
	    ParameterProperties::NONE,
	    SG_OPTIONS(
	        ST_AUTO, ST_CPLEX, ST_GLPK, ST_NEWTON, ST_DIRECT, ST_ELASTICNET,
	        ST_BLOCK_NORM));
}

CMachine::~CMachine()
{
	SG_UNREF(m_labels);
}

bool CMachine::train(CFeatures* data)
{
	if (train_require_labels())
	{
		if (m_labels == NULL)
			error("{}@{}: No labels given", get_name(), fmt::ptr(this));

		m_labels->ensure_valid(get_name());
	}

	auto sub = connect_to_signal_handler();
	bool result = false;

	if (support_feature_dispatching())
	{
		require(data != NULL, "Features not provided!");
		require(
		    data->get_num_vectors() == m_labels->get_num_labels(),
		    "Number of training vectors ({}) does not match number of "
		    "labels ({})\n",
		    data->get_num_vectors(), m_labels->get_num_labels());

		if (support_dense_dispatching() && data->get_feature_class() == C_DENSE)
			result = train_dense(data);
		else if (
		    support_string_dispatching() &&
		    data->get_feature_class() == C_STRING)
			result = train_string(data);
		else
			error("Training with {} is not implemented!", data->get_name());
	}
	else
		result = train_machine(data);

	sub.unsubscribe();
	reset_computation_variables();

	return result;
}

void CMachine::set_labels(CLabels* lab)
{
    if (lab != NULL)
        if (!is_label_valid(lab))
            error("Invalid label for {}", get_name());

	SG_REF(lab);
	SG_UNREF(m_labels);
	m_labels = lab;
}

CLabels* CMachine::get_labels()
{
	SG_REF(m_labels);
	return m_labels;
}

void CMachine::set_max_train_time(float64_t t)
{
	m_max_train_time = t;
}

float64_t CMachine::get_max_train_time()
{
	return m_max_train_time;
}

EMachineType CMachine::get_classifier_type()
{
	return CT_NONE;
}

void CMachine::set_solver_type(ESolverType st)
{
	m_solver_type = st;
}

ESolverType CMachine::get_solver_type()
{
	return m_solver_type;
}

CLabels* CMachine::apply(CFeatures* data)
{
	SG_DEBUG("entering {}::apply({} at {})\n",
			get_name(), data ? data->get_name() : "NULL", fmt::ptr(data));

	CLabels* result=NULL;

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
			error("Unknown problem type");
			break;
	}

	SG_DEBUG("leaving {}::apply({} at {})\n",
			get_name(), data ? data->get_name() : "NULL", fmt::ptr(data));

	return result;
}

CBinaryLabels* CMachine::apply_binary(CFeatures* data)
{
	error("This machine does not support apply_binary()\n");
	return NULL;
}

CRegressionLabels* CMachine::apply_regression(CFeatures* data)
{
	error("This machine does not support apply_regression()\n");
	return NULL;
}

CMulticlassLabels* CMachine::apply_multiclass(CFeatures* data)
{
	error("This machine does not support apply_multiclass()\n");
	return NULL;
}

CStructuredLabels* CMachine::apply_structured(CFeatures* data)
{
	error("This machine does not support apply_structured()\n");
	return NULL;
}

CLatentLabels* CMachine::apply_latent(CFeatures* data)
{
	error("This machine does not support apply_latent()\n");
	return NULL;
}
