/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */

#ifndef _FEATUREDISPATCHCRTP_H__
#define _FEATUREDISPATCHCRTP_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
	class CFeatures;
	class CLabels;
	#define IGNORE_IN_CLASSLIST
	
	template <class P, class T>
	IGNORE_IN_CLASSLIST class CDenseFeaturesDispatch : public T
	{
	public:
		/** Default constructor */
		CDenseFeaturesDispatch() : T()
		{
		}

		virtual ~CDenseFeaturesDispatch()
		{
		}

	protected:
		virtual bool train_dense(CFeatures* data)
		{
			REQUIRE(data->get_num_vectors() == this->m_labels->get_num_labels(), "Number of training vectors (%d) does not match number of labels (%d)\n"	
		, data->get_num_vectors(), this->m_labels->get_num_labels())

			//check for type of CFeatures, then call the appropriate template method
			auto obj = this->template as<P>();
			if(data->get_feature_type() == F_DREAL)
				return obj->template train_machine_templated<float64_t>(data->as<CDenseFeatures<float64_t>>());
			else if(data->get_feature_type() == F_SHORTREAL)
				return obj->template train_machine_templated<float32_t>(data->as<CDenseFeatures<float32_t>>());
			else if(data->get_feature_type() == F_LONGREAL)
				return obj->template train_machine_templated<floatmax_t>(data->as<CDenseFeatures<floatmax_t>>());
			else	
			SG_SERROR("Feature-type (%d) must be of type F_SHORTREAL (%d), F_DREAL (%d) or F_LONGREAL (%d).\n", 	
				data->get_feature_type(), F_SHORTREAL, F_DREAL, F_LONGREAL)	
				
			return false;
		}
		
		virtual bool support_features_dispatching(){ return true; }
	};
	
	template<class P, class T>
	IGNORE_IN_CLASSLIST class CStringFeaturesDispatch : public T
	{
	public:
		/** Default constructor */
		CStringFeaturesDispatch() : T()
		{
		}

		virtual ~CStringFeaturesDispatch()
		{
		}

	protected:
		virtual bool train_string(CFeatures* data)
		{
			REQUIRE(data->get_num_vectors() == this->m_labels->get_num_labels(), "Number of training vectors (%d) does not match number of labels (%d)\n"	
		, data->get_num_vectors(), this->m_labels->get_num_labels())

			//check for type of CFeatures, then call the appropriate template method
			auto obj = this->template as<P>();
			if(data->get_feature_type() == F_DREAL)
				return obj->template train_machine_templated<float64_t>(data->as<CStringFeatures<float64_t>>());
			else if(data->get_feature_type() == F_SHORTREAL)
				return obj->template train_machine_templated<float32_t>(data->as<CStringFeatures<float32_t>>());
			else if(data->get_feature_type() == F_LONGREAL)
				return obj->template train_machine_templated<floatmax_t>(data->as<CStringFeatures<floatmax_t>>());
			else	
			SG_SERROR("Feature-type (%d) must be of type F_SHORTREAL (%d), F_DREAL (%d) or F_LONGREAL (%d).\n", 	
				data->get_feature_type(), F_SHORTREAL, F_DREAL, F_LONGREAL)	
				
			return false;
		}
		
		virtual bool support_features_dispatching(){ return true; }
	};
}
#endif
