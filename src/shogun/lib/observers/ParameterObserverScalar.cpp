/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/observers/ParameterObserverScalar.h>

using namespace shogun;

ParameterObserverScalar::ParameterObserverScalar()
    : ParameterObserverTensorBoard()
{
}

ParameterObserverScalar::ParameterObserverScalar(
    std::vector<std::string>& parameters,std::vector<ParameterProperties>& properties)
    : ParameterObserverTensorBoard(parameters, properties)
{
}

ParameterObserverScalar::ParameterObserverScalar(
    const std::string& filename, std::vector<std::string>& parameters, std::vector<ParameterProperties>& properties)
    : ParameterObserverTensorBoard(filename, parameters, properties)
{
}

ParameterObserverScalar::~ParameterObserverScalar()
{
}

void ParameterObserverScalar::on_next_impl(const TimedObservedValue& value)
{
	auto node_name = std::string("node");
	auto format = TBOutputFormat();
	auto event_value = format.convert_scalar(value, node_name);
	m_writer.writeEvent(event_value);
}

void ParameterObserverScalar::on_error(std::exception_ptr)
{
}

void ParameterObserverScalar::on_complete()
{
}

ParameterObserverScalar::ParameterObserverScalar(std::vector<std::string> &parameters) : ParameterObserverTensorBoard(
		parameters) {

}

ParameterObserverScalar::ParameterObserverScalar(std::vector<ParameterProperties> &properties)
		: ParameterObserverTensorBoard(properties) {

}

#endif // HAVE_TFLOGGER
