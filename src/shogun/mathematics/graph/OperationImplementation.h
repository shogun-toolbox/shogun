#ifndef OPERATIONIMPLEMENTATION_H_
#define OPERATIONIMPLEMENTATION_H_

#include <shogun/mathematics/graph/LinalgNodes.h>

namespace shogun {
IGNORE_IN_CLASSLIST class OperationImpl
{
public:
	OperationImpl(const std::shared_ptr<Node>& abstract_node): m_abstract_node(abstract_node) {}

protected:
	std::shared_ptr<Node> m_abstract_node;
};
}

#endif