/*
* Written (W) 2019 Giovanni De Toni
*/

#include <shogun/base/Parameter.h>
#include <shogun/lib/parameter_observers/ObservedValue.h>

using namespace shogun;

ObservedValue::ObservedValue(int64_t step, char * name)
	: CSGObject(), m_step(step), m_name(name)
{
	SG_ADD(&step, "step", "Step", ParameterProperties::NONE);
	SG_ADD(name, "name", "Name of the observed value", ParameterProperties::NONE);
}
