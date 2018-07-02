/*
* Written (W) 2018 Giovanni De Toni
*/

#include <shogun/io/SGIO.h>
#include <shogun/lib/parameter_observers/ParameterObserverLogger.h>

using namespace shogun;

CParameterObserverLogger::CParameterObserverLogger() : ParameterObserverInterface()
{
	m_type = LOGGER;
}

CParameterObserverLogger::~CParameterObserverLogger() {}

void CParameterObserverLogger::on_next(const TimedObservedValue &value)
{
	CHECK_OBSERVED_VALUE_TYPE(value.first.get_type());
	// Just get the value for now. It does nothing else.
	//SG_SPRINT("Value observed");
}

void CParameterObserverLogger::on_error(std::exception_ptr ptr) {}
void CParameterObserverLogger::on_complete() {}
