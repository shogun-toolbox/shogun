/*
* Written (W) 2017 Giovanni De Toni
*/

#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/ParameterObserverScalar.h>

using namespace shogun;

ParameterObserverScalar::ParameterObserverScalar()
    : ParameterObserverTensorBoard()
{
}

ParameterObserverScalar::ParameterObserverScalar(
    std::vector<std::string>& parameters)
    : ParameterObserverTensorBoard(parameters)
{
}

ParameterObserverScalar::ParameterObserverScalar(
    const std::string& filename, std::vector<std::string>& parameters)
    : ParameterObserverTensorBoard(filename, parameters)
{
}

ParameterObserverScalar::~ParameterObserverScalar()
    : ~ParameterObserverTensorBoard()
{
}

void ParameterObserverScalar::on_next(const ObservedValue& value)
{
	auto node_name = std::string("node");
	auto format = TBOutputFormat();
	auto event_value =
	    format.convert_scalar(value.first, value.second, node_name);
	m_writer.writeEvent(event_value);
}

void ParameterObserverScalar::on_error(std::exception_ptr)
{
}

void ParameterObserverScalar::on_complete()
{
}

bool ParameterObserverScalar::filter(const std::string& param)
{
	if (m_parameters.size() == 0)
		return true;

	for (auto v : m_parameters)
		if (v == param)
			return true;
	return false;
}
