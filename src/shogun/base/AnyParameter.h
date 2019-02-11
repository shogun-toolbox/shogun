/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Gil Hoben
 */

#ifndef __ANYPARAMETER_H__
#define __ANYPARAMETER_H__

#include <shogun/lib/any.h>
#include <shogun/lib/bitmask_operators.h>

#include <string>

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

	/** parameter properties */
	enum class ParameterProperties
	{
		NONE = 0,
		HYPER = 1u << 0,
		GRADIENT = 1u << 1,
		MODEL = 1u << 2,
		AUTO = 1u << 10
	};

	enableEnumClassBitmask(ParameterProperties);

	/** @brief Class AnyParameterProperties keeps track of of parameter meta
	 * information, such as properties and descriptions The parameter properties
	 * can be either true or false. These properties describe if a parameter is
	 * for example a hyperparameter or if it has a gradient.
	 */
	class AnyParameterProperties
	{
	public:
		/** Default constructor where all parameter properties are false
		 */
		AnyParameterProperties()
		    : m_description("No description given"),
		      m_model_selection(MS_NOT_AVAILABLE),
		      m_gradient(GRADIENT_NOT_AVAILABLE),
		      m_attribute_mask(ParameterProperties::NONE)
		{
		}
		/** Constructor
		 * @param description parameter description
		 * @param hyperparameter set to true for parameters that determine
		 * how training is performed, e.g. regularisation parameters
		 * @param gradient set to true for parameters required for gradient
		 * updates
		 * @param model set to true for parameters used in inference, e.g.
		 * weights and bias
		 * */
		AnyParameterProperties(
		    std::string description,
		    EModelSelectionAvailability hyperparameter = MS_NOT_AVAILABLE,
		    EGradientAvailability gradient = GRADIENT_NOT_AVAILABLE,
		    bool model = false)
		    : m_description(description), m_model_selection(hyperparameter),
		      m_gradient(gradient)
		{
			m_attribute_mask = ParameterProperties::NONE;
			if (hyperparameter)
				m_attribute_mask |= ParameterProperties::HYPER;
			if (gradient)
				m_attribute_mask |= ParameterProperties::GRADIENT;
			if (model)
				m_attribute_mask |= ParameterProperties::MODEL;
		}
		/** Mask constructor
		 * @param description parameter description
		 * @param attribute_mask mask encoding parameter properties
		 * */
		AnyParameterProperties(
		    std::string description, ParameterProperties attribute_mask)
		    : m_description(description), m_attribute_mask(attribute_mask)
		{
		}
		/** Copy contructor */
		AnyParameterProperties(const AnyParameterProperties& other)
		    : m_description(other.m_description),
		      m_model_selection(other.m_model_selection),
		      m_gradient(other.m_gradient),
		      m_attribute_mask(other.m_attribute_mask)
		{
		}
		const std::string& get_description() const
		{
			return m_description;
		}
		EModelSelectionAvailability get_model_selection() const
		{
			return static_cast<EModelSelectionAvailability>(
			    static_cast<int32_t>(
			        m_attribute_mask & ParameterProperties::HYPER) > 0);
		}
		EGradientAvailability get_gradient() const
		{
			return static_cast<EGradientAvailability>(
			    static_cast<int32_t>(
			        m_attribute_mask & ParameterProperties::GRADIENT) > 0);
		}
		bool get_model() const
		{
			return static_cast<bool>(
			    m_attribute_mask & ParameterProperties::MODEL);
		}
		bool has_property(ParameterProperties other) const
		{
			return static_cast<bool>(m_attribute_mask & other);
		}
		bool compare_mask(ParameterProperties other) const
		{
			return m_attribute_mask == other;
		}
		void remove_property(ParameterProperties other)
		{
			m_attribute_mask &= ~other;
		}

	private:
		std::string m_description;
		EModelSelectionAvailability m_model_selection;
		EGradientAvailability m_gradient;
		ParameterProperties m_attribute_mask;
	};

	class AnyParameter
	{
	public:
		AnyParameter() : m_value(), m_properties()
		{
		}
		explicit AnyParameter(const Any& value) : m_value(value), m_properties()
		{
			m_init_function = [& m_value = m_value]() { return m_value; };
		}
		AnyParameter(const Any& value, AnyParameterProperties properties)
		    : m_value(value), m_properties(properties)
		{
			m_init_function = [& m_value = m_value]() { return m_value; };
		}
		AnyParameter(
		    const Any& value, AnyParameterProperties properties,
		    std::function<Any()> lambda_)
		    : m_value(value), m_properties(properties),
		      m_init_function(std::move(lambda_))
		{
		}
		AnyParameter(const AnyParameter& other)
		    : m_value(other.m_value), m_properties(other.m_properties),
		      m_init_function(other.m_init_function)
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

		std::function<Any()> get_init_function() const
		{
			return m_init_function;
		}

		/** Equality operator which compares value but not properties.
		 * @return true if value of other parameter equals own */
		inline bool operator==(const AnyParameter& other) const
		{
			return m_value == other.get_value();
		}

		/** @see operator==() */
		inline bool operator!=(const AnyParameter& other) const
		{
			return !(*this == other);
		}

	private:
		Any m_value;
		AnyParameterProperties m_properties;
		std::function<Any()> m_init_function;
	};
} // namespace shogun

#endif
