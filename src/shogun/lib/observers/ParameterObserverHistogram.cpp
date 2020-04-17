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
#include <shogun/lib/observers/ParameterObserverHistogram.h>

using namespace shogun;

ParameterObserverHistogram::ParameterObserverHistogram()
    : ParameterObserverTensorBoard()
{
}

ParameterObserverHistogram::ParameterObserverHistogram(
    std::vector<std::string>& parameters,
    std::vector<ParameterProperties>& properties)
    : ParameterObserverTensorBoard(parameters, properties)
{
}

ParameterObserverHistogram::ParameterObserverHistogram(
    const std::string& filename, std::vector<std::string>& parameters,
    std::vector<ParameterProperties>& properties)
    : ParameterObserverTensorBoard(filename, parameters, properties)
{
}

ParameterObserverHistogram::~ParameterObserverHistogram()
{
}

void ParameterObserverHistogram::on_next_impl(const TimedObservedValue& value)
{
	auto node_name = std::string("node");
	auto format = TBOutputFormat();
	auto event_value = format.convert_vector(value, node_name);
	m_writer.writeEvent(event_value);
}

void ParameterObserverHistogram::on_error(std::exception_ptr)
{
}

void ParameterObserverHistogram::on_complete()
{
}

ParameterObserverHistogram::ParameterObserverHistogram(
    std::vector<std::string>& parameters)
    : ParameterObserverTensorBoard(parameters)
{
}

ParameterObserverHistogram::ParameterObserverHistogram(
    std::vector<ParameterProperties>& properties)
    : ParameterObserverTensorBoard(properties)
{
}

#endif // HAVE_TFLOGGER
