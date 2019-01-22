/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */

#ifndef _FEATUREDISPATCHCRTP_H__
#define _FEATUREDISPATCHCRTP_H__

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	class CFeatures;
	class CLabels;
#define IGNORE_IN_CLASSLIST

	template <class P, class T>
	IGNORE_IN_CLASSLIST class CDenseRealDispatch : public T
	{
	public:
		/** Default constructor */
		CDenseRealDispatch() : T()
		{
		}

		virtual ~CDenseRealDispatch()
		{
		}

	protected:
		virtual void train_dense(CFeatures* features, CLabels* labels)
		{
			auto this_casted = this->template as<P>();
			switch (features->get_feature_type())
			{
			case F_DREAL:
			{
				this_casted->template train_machine_templated<float64_t>(
				    features->as<CDenseFeatures<float64_t>>(), labels);
				return;
			}
			case F_SHORTREAL:
			{
				this_casted->template train_machine_templated<float32_t>(
				    features->as<CDenseFeatures<float32_t>>(), labels);
				return;
			}
			case F_LONGREAL:
			{
				this_casted
				    ->template train_machine_templated<floatmax_t>(
				        features->as<CDenseFeatures<floatmax_t>>(), labels);
				return;
			}
			default:
				SG_SERROR(
				    "Training with %s of provided type %s is not "
				    "possible!",
				    features->get_name(),
				    feature_type(features->get_feature_type()).c_str());
			}
		}

		virtual bool support_feature_dispatching()
		{
			return true;
		}

		virtual bool support_dense_dispatching()
		{
			return true;
		}
	};

	template <class P, class T>
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
		virtual void train_string(CFeatures* data)
		{
			auto this_casted = this->template as<P>();
			switch (data->get_feature_type())
			{
			case F_BYTE:
			{
				this_casted->template train_machine_templated<uint8_t>(
				    data->as<CStringFeatures<uint8_t>>());
				return;
			}
			case F_CHAR:
			{
				this_casted->template train_machine_templated<char>(
				    data->as<CStringFeatures<char>>());
				return;
			}
			case F_WORD:
			{
				this_casted->template train_machine_templated<uint16_t>(
				    data->as<CStringFeatures<uint16_t>>());
				return;
			}
			default:
				SG_SERROR(
				    "Training with %s of provided type %s is "
				    "not possible!",
				    data->get_name(),
				    feature_type(data->get_feature_type()).c_str());
			}
		}

		virtual bool support_feature_dispatching()
		{
			return true;
		}

		virtual bool support_string_dispatching()
		{
			return true;
		}
	};
}
#endif
