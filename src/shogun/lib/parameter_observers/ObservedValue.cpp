/*
* Written (W) 2019 Giovanni De Toni
*/

#include <shogun/base/Parameter.h>
#include <shogun/lib/parameter_observers/ObservedValue.h>

using namespace shogun;

ObservedValue::ObservedValue(int64_t step, std::string name)
	: CSGObject(), m_step(step), m_name(name)
{
	SG_ADD(&m_step, "step", "Step", ParameterProperties::NONE);
	this->watch_param("name", &m_name,
					  AnyParameterProperties("Name of the observed value", ParameterProperties::NONE));
}
