/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Gil Hoben
 */

#ifndef __ANYPARAMETER_H__
#define __ANYPARAMETER_H__

#include <shogun/lib/any.h>

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
	/** @brief Class AnyParameterProperties keeps track of parameter properties.
	 * The parameter properties can be either true or false.
	 * These properties describe if a parameter is for example a hyperparameter
	 * or if it has a gradient.
	 */
	class AnyParameterProperties
	{
	public:
		static const int32_t HYPERPARAMETER = 0x00000001;
		static const int32_t GRADIENT_PARAM = 0x00000010;
		static const int32_t MODEL_PARAM = 0x00000100;

		/** Default constructor where all parameter properties are false
		 */
		AnyParameterProperties() : m_description(), m_mask_attribute(0x00000000)
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
		    EModelSelectionAvailability model_selection = MS_NOT_AVAILABLE,
		    EGradientAvailability gradient = GRADIENT_NOT_AVAILABLE,
		    bool hyperparameter = false)
		    : m_description(description), m_model_selection(model_selection),
		      m_gradient(gradient)
		{
			m_mask_attribute = 0x00000000;
			if (model_selection)
				m_mask_attribute |= MODEL_PARAM;
			if (gradient)
				m_mask_attribute |= GRADIENT_PARAM;
			if (hyperparameter)
				m_mask_attribute |= HYPERPARAMETER;
		}
		/** Mask constructor
		 * @param description parameter description
		 * @param attribute_mask mask encoding parameter properties
		 * */
		    : m_description(description)
		{
			m_mask_attribute = mask_attribute;
		}
		/** Copy contructor */
		AnyParameterProperties(const AnyParameterProperties& other)
		    : m_description(other.m_description),
		      m_model_selection(other.m_model_selection),
		      m_gradient(other.m_gradient),
		      m_mask_attribute(other.m_mask_attribute)
		{
		}
		const std::string& get_description() const
		std::string get_description() const
		{
			return m_description;
		}
		EModelSelectionAvailability get_model_selection() const
		{
			EModelSelectionAvailability return_val;
			if (m_mask_attribute & HYPERPARAMETER)
				return_val = EModelSelectionAvailability::MS_AVAILABLE;
			else
				return_val = EModelSelectionAvailability::MS_NOT_AVAILABLE;
			return return_val;
		}

		EGradientAvailability get_gradient() const
		{
			EGradientAvailability return_val;
			if (m_mask_attribute & GRADIENT_PARAM)
				return_val = EGradientAvailability::GRADIENT_AVAILABLE;
			else
				return_val = EGradientAvailability::GRADIENT_NOT_AVAILABLE;
			return return_val;
		}

		bool get_hyperparameter() const
		{
			bool return_val;
			if (m_mask_attribute & HYPERPARAMETER)
				return_val = true;
			else
				return_val = false;
			return return_val;
		}

	private:
		std::string m_description;
		EModelSelectionAvailability m_model_selection;
		EGradientAvailability m_gradient;
		int32_t m_attribute_mask;
		int32_t m_mask_attribute;
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
