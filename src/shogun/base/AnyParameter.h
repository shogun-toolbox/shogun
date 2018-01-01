#ifndef __ANYPARAMETER_H__
#define __ANYPARAMETER_H__

#include <shogun/lib/any.h>

namespace shogun
{

	/** model selection availability */
	enum EModelSelectionAvailability
	{
		MS_NOT_AVAILABLE = 0,
		MS_AVAILABLE = 1,
	};

	/** gradient availability */
	enum EGradientAvailability
	{
		GRADIENT_NOT_AVAILABLE = 0,
		GRADIENT_AVAILABLE = 1
	};

	class AnyParameterProperties
	{
	public:
		AnyParameterProperties()
		    : m_model_selection(MS_NOT_AVAILABLE),
		      m_gradient(GRADIENT_NOT_AVAILABLE)
		{
		}
		AnyParameterProperties(
		    EModelSelectionAvailability model_selection,
		    EGradientAvailability gradient)
		    : m_model_selection(model_selection), m_gradient(gradient)
		{
		}
		AnyParameterProperties(const AnyParameterProperties& other)
		    : m_model_selection(other.m_model_selection),
		      m_gradient(other.m_gradient)
		{
		}

		EModelSelectionAvailability get_model_selection() const
		{
			return m_model_selection;
		}

		EGradientAvailability get_gradient() const
		{
			return m_gradient;
		}

	private:
		EModelSelectionAvailability m_model_selection;
		EGradientAvailability m_gradient;
	};

	class AnyParameter
	{
	public:
		AnyParameter() : m_value(), m_properties()
		{
		}
		explicit AnyParameter(const Any& value) : m_value(value), m_properties()
		{
		}
		AnyParameter(const Any& value, AnyParameterProperties properties)
		    : m_value(value), m_properties(properties)
		{
		}
		AnyParameter(const AnyParameter& other)
		    : m_value(other.m_value), m_properties(other.m_properties)
		{
		}

		Any get_value() const
		{
			return m_value;
		}

		void set_value(const Any& value)
		{
			m_value = value;
		}

		AnyParameterProperties get_properties() const
		{
			return m_properties;
		}

	private:
		Any m_value;
		AnyParameterProperties m_properties;
	};
}

#endif
