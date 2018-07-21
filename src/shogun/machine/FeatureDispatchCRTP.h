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
			auto obj = this->template as<P>();
			switch (data->get_feature_type())
			{
			case F_DREAL:
				return obj->template train_machine_templated<float64_t>(
				    data->as<CDenseFeatures<float64_t>>());
			case F_SHORTREAL:
				return obj->template train_machine_templated<float32_t>(
				    data->as<CDenseFeatures<float32_t>>());
			case F_LONGREAL:
				return obj->template train_machine_templated<floatmax_t>(
				    data->as<CDenseFeatures<floatmax_t>>());
			default:
				SG_SERROR(
				    "Training with DenseFeatures of provided %s type is not "
				    "possible!",
				    feature_type(data->get_feature_type()));
			}
			return false;
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
		virtual bool train_string(CFeatures* data)
		{
			auto obj = this->template as<P>();
			switch (data->get_feature_type())
			{
			case F_BYTE:
				return obj->template train_machine_templated<uint8_t>(
				    data->as<CStringFeatures<uint8_t>>());
			case F_CHAR:
				return obj->template train_machine_templated<char>(
				    data->as<CStringFeatures<char>>());
			case F_WORD:
				return obj->template train_machine_templated<uint16_t>(
				    data->as<CStringFeatures<uint16_t>>());
			default:
				SG_SERROR(
				    "Training with StringFeatures of provided %s type is "
				    "not possible!",
				    feature_type(data->get_feature_type()));
			}
			return false;
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
