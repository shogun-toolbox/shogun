#include "distributions/hmm/GHMM.h"

bool CGHMM::train()
{
	return false;
}

INT CGHMM::get_num_model_parameters()
{
	return 0;
}

DREAL CGHMM::get_log_model_parameter(INT param_num)
{
	return 0;
}

DREAL CGHMM::get_log_derivative(INT param_num, INT num_example)
{
	return 0;
}

DREAL CGHMM::get_log_likelihood_example(INT num_example)
{
	return 0;
}

