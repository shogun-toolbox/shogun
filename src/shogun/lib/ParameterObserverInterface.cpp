#include <shogun/lib/ParameterObserverInterface.h>

using namespace shogun;

ParameterObserverInterface::ParameterObserverInterface()
: m_parameters(), m_writer("shogun")
{
}

ParameterObserverInterface::ParameterObserverInterface(
    std::vector<std::string>& parameters)
    : m_parameters(parameters), m_writer("shogun")
{
}

ParameterObserverInterface::ParameterObserverInterface(
    const std::string& filename, std::vector<std::string>& parameters)
    : m_parameters(parameters), m_writer(filename.c_str())
{
}

ParameterObserverInterface::~ParameterObserverInterface()
{
}
