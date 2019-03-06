#include <shogun/lib/parameter_observers/ObservedValue.h>

using namespace shogun;

ObservedValue::ObservedValue(int64_t step, std::string name, Any value)
		    : m_step(step), m_name(name), m_value(value)
{
}

ObservedValue::~ObservedValue() {}