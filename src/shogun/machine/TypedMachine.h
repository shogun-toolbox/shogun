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
				if (!data)
				{	
					REQUIRE(this->features, "No features provided.\n")	
					REQUIRE(this->features->get_feature_class() == C_DENSE,	
					"Feature-class (%d) must be of type C_DENSE (%d)\n", this->features->get_feature_class(), C_DENSE)	
					
					data = this->features;	
				}	
		else	
			REQUIRE(data->get_feature_class() == C_DENSE,	
				"Feature-class must be of type C_DENSE (%d)\n", data->get_feature_class(), C_DENSE)	
		
			REQUIRE(data->get_num_vectors() == this->m_labels->get_num_labels(), "Number of training vectors (%d) does not match number of labels (%d)\n"	
		, data->get_num_vectors(), this->m_labels->get_num_labels())
			
			//check for type of CFeatures, then call the appropriate template method
			if(data->get_feature_type() == F_DREAL)
				return this->template as<P>()->template train_machine_templated<float64_t>(data->as<CDenseFeatures<float64_t>>());
			else if(data->get_feature_type() == F_SHORTREAL)
				return this->template as<P>()->template train_machine_templated<float32_t>(data->as<CDenseFeatures<float32_t>>());
			else if(data->get_feature_type() == F_LONGREAL)
				return this->template as<P>()->template train_machine_templated<floatmax_t>(data->as<CDenseFeatures<floatmax_t>>());
			else	
			SG_SERROR("Feature-type (%d) must be of type F_SHORTREAL (%d), F_DREAL (%d) or F_LONGREAL (%d).\n", 	
				data->get_feature_type(), F_SHORTREAL, F_DREAL, F_LONGREAL)	
	
			return false;
		}
		
		virtual bool support_dispatching()=0;
	};
}
#endif
