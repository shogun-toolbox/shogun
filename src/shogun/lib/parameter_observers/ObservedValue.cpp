#include <shogun/lib/parameter_observers/ObservedValue.h>

using namespace shogun;

ObservedValue::ObservedValue(int64_t step, std::string& name, Any value)
		    : m_step(step), m_name(name), m_value(value)
{
	SG_ADD(&m_step, "step", "Observation step", ParameterProperties::NONE);
	SG_ADD(&m_name, "name", "Observed value name", ParameterProperties::NONE);
	SG_ADD(&m_value, "value", "Observed value", ParameterProperties::NONE);
}

ObservedValue::~ObservedValue() {}