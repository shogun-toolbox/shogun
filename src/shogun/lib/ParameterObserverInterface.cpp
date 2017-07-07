#include <shogun/lib/ParameterObserverInterface.h>

using namespace shogun;

ParameterObserverInterface::ParameterObserverInterface() : m_parameters()
{
}

ParameterObserverInterface::ParameterObserverInterface(
    std::vector<std::string>& parameters)
    : m_parameters(parameters)
{
}

ParameterObserverInterface::ParameterObserverInterface(
    const std::string& filename, std::vector<std::string>& parameters)
    : m_parameters(parameters)
{
}

ParameterObserverInterface::~ParameterObserverInterface()
{
}
