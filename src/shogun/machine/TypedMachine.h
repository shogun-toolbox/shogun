/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */

#ifndef _TYPEDMACHINE_H__
#define _TYPEDMACHINE_H__

#include <shogun/lib/config.h>

#include <shogun/base/progress.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	class CFeatures;
	class CLabels;
	
	template <class P, class T>
	class CTypedMachine : public T
	{
	public:
		/** Default constructor */
		CTypedMachine() : T()
		{
		}

		virtual ~CTypedMachine()
		{
		}

	protected:
		virtual bool train_machine(CFeatures* data = NULL)
		{
			//check for type of CFeatures, then call the appropriate template method
			P::set_compute_bias(true);
			if(data->get_feature_type() == F_DREAL)
				return P::template train_machine_templated<float64_t>(data->as<CDenseFeatures<float64_t>>());
			else if(data->get_feature_type() == F_SHORTREAL)
				return P::template train_machine_templated<float32_t>(data->as<CDenseFeatures<float32_t>>());
			else if(data->get_feature_type() == F_LONGREAL)
				return P::template train_machine_templated<floatmax_t>(data->as<CDenseFeatures<floatmax_t>>());

			return false;
		}
	};
}
#endif
