/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/observers/ParameterObserverTensorBoard.h>

using namespace shogun;

ParameterObserverTensorBoard::ParameterObserverTensorBoard()
    : ParameterObserver(), m_writer("shogun")
{
	m_writer.init();
}

ParameterObserverTensorBoard::ParameterObserverTensorBoard(
    std::vector<std::string>& parameters,
    std::vector<ParameterProperties>& properties)
    : ParameterObserver(parameters, properties), m_writer("shogun")
{
	m_writer.init();
}

ParameterObserverTensorBoard::ParameterObserverTensorBoard(
    const std::string& filename, std::vector<std::string>& parameters,
    std::vector<ParameterProperties>& properties)
    : ParameterObserver(filename, parameters, properties),
      m_writer(filename.c_str())
{
	m_writer.init();
}

ParameterObserverTensorBoard::~ParameterObserverTensorBoard()
{
	m_writer.flush();
	m_writer.close();
}

ParameterObserverTensorBoard::ParameterObserverTensorBoard(
    std::vector<std::string>& parameters)
    : ParameterObserver(parameters), m_writer("shogun")
{
	m_writer.init();
}

ParameterObserverTensorBoard::ParameterObserverTensorBoard(
    std::vector<ParameterProperties>& properties)
    : ParameterObserver(properties), m_writer("shogun")
{
	m_writer.init();
}

#endif // HAVE_TFLOGGER
