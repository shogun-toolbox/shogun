#ifndef _CGHMM_H___
#define _CGHMM_H___

#include "lib/Mathmatics.h"
#include "features/Features.h"
#include "distributions/Distribution.h"


class CGHMM : CDistribution
{
	public:
		virtual bool train();
		virtual INT get_num_model_parameters();
		virtual REAL get_log_model_parameter(INT param_num);
		virtual REAL get_log_derivative(INT param_num, INT num_example);
		virtual REAL get_log_likelihood_example(INT num_example);
};
#endif
