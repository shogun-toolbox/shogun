#ifndef _HISTOGRAM_H___
#define _HISTOGRAM_H___

#include "features/WordFeatures.h"
#include "distributions/Distribution.h"

class CHistogram : private CDistribution
{
	public:
		CHistogram();
		~CHistogram();

		virtual bool train();

		virtual inline INT get_num_model_parameters()
		{
			return (1<<16);
		}
		virtual REAL get_log_model_parameter(INT param_num);

		virtual REAL get_log_derivative(INT num_example, INT num_param);
		virtual REAL get_log_likelihood_example(INT num_example);

	protected:
		REAL* hist;
		CWordFeatures* features;
};
#endif
