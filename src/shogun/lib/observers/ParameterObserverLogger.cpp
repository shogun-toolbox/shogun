/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#include <shogun/io/visitors/ToStringVisitor.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/observers/ParameterObserverLogger.h>
#include <shogun/lib/type_case.h>

using namespace shogun;

CParameterObserverLogger::CParameterObserverLogger()
{
}

CParameterObserverLogger::CParameterObserverLogger(
    std::vector<std::string>& parameters)
    : ParameterObserver(parameters)
{
}

CParameterObserverLogger::~CParameterObserverLogger()
{
}

void CParameterObserverLogger::on_error(std::exception_ptr ptr)
{
}

void CParameterObserverLogger::on_complete()
{
}

void CParameterObserverLogger::on_next_impl(const TimedObservedValue& value)
{

	auto name = value.first->get<std::string>("name");
	auto any_val = value.first->get_any();

	std::stringstream stream;
	ToStringVisitor visitor(&stream);

	if (any_val.visitable())
	{
		any_val.visit(&visitor);
	} else {
		stream << "{function}";
	}

	SG_PRINT(
			"[%lu] \"%s\" = %s\n",
			convert_to_millis(value.second), name.c_str(),
			stream.str().c_str());
}
