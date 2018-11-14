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
		HYPER = 1u << 0,
		GRADIENT = 1u << 1,
		MODEL = 1u << 2
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
		static const int32_t HYPER = 1u << 0;
		static const int32_t GRADIENT = 1u << 1;
		static const int32_t MODEL = 1u << 2;

		/** Default constructor where all parameter properties are false
		 */
		AnyParameterProperties()
		    : m_description("No description given"), m_attribute_mask(0)
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
			m_attribute_mask = (hyperparameter << 0 & HYPER) |
			                   (gradient << 1 & GRADIENT) |
			                   (model << 2 & MODEL);
		}
		/** Mask constructor
		 * @param description parameter description
		 * @param attribute_mask mask encoding parameter properties
		 * */
		AnyParameterProperties(std::string description, int32_t attribute_mask)
		    : m_description(description)
		{
			m_attribute_mask = attribute_mask;
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
			    (m_attribute_mask & HYPER) > 0);
		}
		EGradientAvailability get_gradient() const
		{
			return static_cast<EGradientAvailability>(
			    (m_attribute_mask & GRADIENT) > 0);
		}
		bool get_model() const
		{
			return static_cast<bool>(m_attribute_mask & MODEL);
		}

	private:
		std::string m_description;
		EModelSelectionAvailability m_model_selection;
		EGradientAvailability m_gradient;
		int32_t m_attribute_mask;
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
	};
} // namespace shogun

#endif
