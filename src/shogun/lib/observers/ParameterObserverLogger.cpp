/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#include <shogun/lib/type_case.h>
#include <shogun/lib/observers/ParameterObserverLogger.h>

using namespace shogun;

ParameterObserverLogger::ParameterObserverLogger() {}

ParameterObserverLogger::ParameterObserverLogger(std::vector<std::string> &parameters) : ParameterObserver(
		parameters) {}

ParameterObserverLogger::~ParameterObserverLogger() {

}

void ParameterObserverLogger::on_error(std::exception_ptr ptr) {

}

void ParameterObserverLogger::on_complete() {

}

void ParameterObserverLogger::on_next_impl(const TimedObservedValue &value) {

	auto name = value.first->get<std::string>("name");
	auto any_val = value.first->get_any();
	auto print_func = [&](auto v){
		SG_PRINT("[%l] Received a value called %s which contains: %s", convert_to_millis(value.second),
				 name, v);
	};

	sg_any_dispatch(any_val, sg_all_typemap, print_func, print_func, print_func);

}
