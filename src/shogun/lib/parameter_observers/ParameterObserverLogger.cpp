/*
* Written (W) 2019 Giovanni De Toni
*/

#include <shogun/lib/parameter_observers/ParameterObserverLogger.h>
//#include <shogun/lib/type_case.h>

using namespace shogun;

ParameterObserverLogger::ParameterObserverLogger() {}

ParameterObserverLogger::ParameterObserverLogger(std::vector<std::string> &parameters)
		: ParameterObserverInterface(parameters) {}

void ParameterObserverLogger::on_next(const TimedObservedValue &value) {

	//auto f_print = [&value](auto _disp_value) {
	//	SG_SPRINT("[%l] Received %s", convert_to_millis(value.second), _disp_value)
	//};
	// Dispatch the value and print it
	//sg_any_dispatch(value.first.get_value(), sg_all_typemap, f_print, f_print, f_print);

	SG_PRINT("[%l] Received an observed value with name \"%s\"",
			 convert_to_millis(value.second), value.first.get_name().c_str());

}

void ParameterObserverLogger::on_error(std::exception_ptr ptr) {

}

void ParameterObserverLogger::on_complete() {

}
