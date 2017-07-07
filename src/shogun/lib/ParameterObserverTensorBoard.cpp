/*
* Written (W) 2017 Giovanni De Toni
*/

#include "ParameterObserverTensorBoard.h"

using namespace shogun;

ParameterObserverTensorBoard::ParameterObserverTensorBoard()
    : ParameterObserverInterface(), m_writer("shogun")
{
	m_writer.init();
}

ParameterObserverTensorBoard::ParameterObserverTensorBoard(
    std::vector<std::string>& parameters)
    : ParameterObserverInterface(parameters), m_writer("shogun")
{
	m_writer.init();
}

ParameterObserverTensorBoard::ParameterObserverTensorBoard(
    const std::string& filename, std::vector<std::string>& parameters)
    : ParameterObserverInterface(parameters), m_writer(filename.c_str())
{
	m_writer.init();
}

ParameterObserverTensorBoard::~ParameterObserverTensorBoard()
    : ~ParameterObserverTensorBoard()
{
	m_writer.flush();
	m_writer.close();
}
