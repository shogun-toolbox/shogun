/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#include <shogun/base/Parameter.h>
#include <shogun/lib/parameter_observers/ObservedValue.h>

using namespace shogun;

ObservedValue::ObservedValue(int64_t step, std::string name)
    : CSGObject(), m_step(step), m_name(name), m_any_value(make_any(nullptr))
{
	SG_ADD(&m_step, "step", "Step");
	this->watch_param(
	    "name", &m_name,
	    AnyParameterProperties("Name of the observed value"));
}
