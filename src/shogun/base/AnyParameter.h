/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Gil Hoben
 */

#ifndef __ANYPARAMETER_H__
#define __ANYPARAMETER_H__

#include <shogun/lib/abstract_auto_init.h>
#include <shogun/lib/any.h>
#include <shogun/lib/bitmask_operators.h>

#include <list>
#include <memory>
#include <string_view>

namespace shogun
{
	/** parameter properties */
	enum class ParameterProperties : uint32_t
	{
		NONE = 0,

		// influence training but not changed in training (e.g. kernel
		// bandwidth), could be learned by model selection
		HYPER = 1u << 0,

		// gradient available/implemented
		GRADIENT = 1u << 1,

		// result/outcome of training
		MODEL = 1u << 2,

		// influence training, but no hyper parameter of model (e.g. batch size,
		// bias used)
		SETTING = 1u << 3,

		// has automatically computed value
		AUTO = 1u << 10,

		READONLY = 1u << 11,

		// an executable function
		RUNFUNCTION = 1u << 12,

		CONSTRAIN = 1u << 13,

		ALL = std::numeric_limits<uint32_t>::max(),
	};

	static const std::list<std::pair<ParameterProperties, std::string>>
	    kParameterPropNames = {
	        {ParameterProperties::NONE, "NONE"},
	        {ParameterProperties::HYPER, "HYPER"},
	        {ParameterProperties::GRADIENT, "GRADIENT"},
	        {ParameterProperties::MODEL, "MODEL"},
	        {ParameterProperties::SETTING, "SETTING"},
	        {ParameterProperties::AUTO, "AUTO"},
	        {ParameterProperties::READONLY, "READONLY"},
	        {ParameterProperties::RUNFUNCTION, "RUNFUNCTION"},
	        {ParameterProperties::CONSTRAIN, "CONSTRAIN"}};

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
		      m_attribute_mask(ParameterProperties::NONE)
		{
		}
		/** Constructor with description and all parameters set to false
		 * @param description parameter description
		 * */
		AnyParameterProperties(
				std::string_view description)
				: m_description(description), m_attribute_mask(ParameterProperties::NONE)
		{
		}
		/** Mask constructor
		 * @param description parameter description
		 * @param attribute_mask mask encoding parameter properties
		 * */
		AnyParameterProperties(
		    std::string_view description, ParameterProperties attribute_mask)
		    : m_description(description), m_attribute_mask(attribute_mask)
		{
		}
		/** Copy contructor */
		AnyParameterProperties(const AnyParameterProperties& other)
		    : m_description(other.m_description),
		      m_attribute_mask(other.m_attribute_mask)
		{
		}
		std::string_view get_description() const
		{
			return m_description;
		}
		bool has_property(const ParameterProperties other) const
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
		std::string to_string() const
		{
			std::stringstream ss;
			ss << "Description: " << m_description.data() << " with attributes: [";
			bool first_attrib = true;
			for (const auto& it : kParameterPropNames)
			{
				if (has_property(it.first))
				{
					ss << (first_attrib ? "" : " | ") << it.second;
					first_attrib = false;
				}
			}
			ss << "]";
			return ss.str();
		}

	private:
		std::string_view m_description;
		ParameterProperties m_attribute_mask;
	};

	class AnyParameter
	{
	public:
		AnyParameter() : m_value(), m_properties()
		{
		}
		explicit AnyParameter(Any&& value)
		    : m_value(std::move(value)), m_properties()
		{
		}
		AnyParameter(Any&& value, const AnyParameterProperties& properties)
		    : m_value(std::move(value)), m_properties(properties)
		{
		}
		AnyParameter(
		    Any&& value, const AnyParameterProperties& properties,
		    std::shared_ptr<params::AutoInit> auto_init)
		    : m_value(std::move(value)), m_properties(properties),
		      m_init_function(std::move(auto_init))
		{
		}
		AnyParameter(
		    Any&& value, const AnyParameterProperties& properties,
		    std::function<std::string(Any)> constrain_function)
		    : m_value(std::move(value)), m_properties(properties),
		      m_constrain_function(std::move(constrain_function))
		{
		}
		AnyParameter(const AnyParameter& other)
		    : m_value(other.m_value), m_properties(other.m_properties),
		      m_init_function(other.m_init_function),
		      m_constrain_function(other.m_constrain_function)
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

		AnyParameterProperties& get_properties()
		{
			return m_properties;
		}

		const AnyParameterProperties& get_properties() const
		{
			return m_properties;
		}

		const std::shared_ptr<params::AutoInit>& get_init_function() const
		{
			return m_init_function;
		}

		const std::function<std::string(Any)>& get_constrain_function() const
		    noexcept
		{
			return m_constrain_function;
		}

		void add_callback_function(std::function<void()>&& method)
		{
			m_callback_functions.push_back(std::move(method));
		}

		const auto& get_callbacks() const
		{
			return m_callback_functions;
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

		inline bool operator<(const AnyParameter& other) const
		{
			// let's assume this assumption is sufficient
			// distinguish AnyParameter objects
			return m_value.hash() < other.m_value.hash();
		}

	private:
		Any m_value;
		AnyParameterProperties m_properties;
		std::shared_ptr<params::AutoInit> m_init_function;
		std::function<std::string(Any)> m_constrain_function;
		std::vector<std::function<void()>> m_callback_functions;
	};
} // namespace shogun

#endif
