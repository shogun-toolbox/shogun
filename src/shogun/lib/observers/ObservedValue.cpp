/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */

#include <shogun/lib/observers/ObservedValue.h>

using namespace shogun;

ObservedValue::ObservedValue(const int64_t step, const std::string& name)
		: CSGObject(), m_step(step), m_name(name), m_any_value(Any())
{
	SG_ADD(&m_step, "step", "Step");
	this->watch_param(
			"name", &m_name, AnyParameterProperties("Name of the observed value"));
}
