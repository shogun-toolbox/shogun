/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#include <shogun/lib/type_case.h>
#include <shogun/lib/observers/ParameterObserverLogger.h>

using namespace shogun;

CParameterObserverLogger::CParameterObserverLogger() {}

CParameterObserverLogger::CParameterObserverLogger(std::vector<std::string> &parameters) : ParameterObserver(
		parameters) {}

CParameterObserverLogger::~CParameterObserverLogger() {

}

void CParameterObserverLogger::on_error(std::exception_ptr ptr) {

}

void CParameterObserverLogger::on_complete() {

}

void CParameterObserverLogger::on_next_impl(const TimedObservedValue &value) {

	auto name = value.first->get<std::string>("name");
	auto any_val = value.first->get_any();

	auto pf_n = [&](auto v){
		SG_PRINT("[%l] Received a value called %s which contains: %s", convert_to_millis(value.second),
				 name.c_str(), std::to_string(v).c_str());
	};

	auto pf_sgvector = [&](auto v){
		//SG_PRINT("[%l] Received a value called %s which contains: %s", convert_to_millis(value.second),
		//		 name.c_str(), v.to_string().c_str());
	};

	auto pf_sgmatrix = [&](auto v){
		//SG_PRINT("[%l] Received a value called %s which contains: %s", convert_to_millis(value.second),
		//		 name.c_str(), v.to_string().c_str());
	};

	sg_any_dispatch(any_val, sg_all_typemap, pf_n, pf_sgvector, pf_sgmatrix);

}
