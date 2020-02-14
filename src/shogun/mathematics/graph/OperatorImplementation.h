#ifndef OPERATIONIMPLEMENTATION_H_
#define OPERATIONIMPLEMENTATION_H_

#include <shogun/mathematics/graph/LinalgNodes.h>

namespace shogun {

IGNORE_IN_CLASSLIST class Operator
{
public:
	Operator() = default;
	virtual ~Operator() {};

	virtual std::string_view get_operator_name() const = 0;

	void operator()()
	{
		if (!m_abstract_node)
			error("Call Operator::build(node), before running Operator.");
		evaluate();
	}
	
	void build(const std::shared_ptr<Node>& node)
	{
		m_abstract_node = node;
	}

	virtual void evaluate() = 0;
	
	protected:
	std::shared_ptr<Node> m_abstract_node;
};

template <typename BackendType>
IGNORE_IN_CLASSLIST class OperatorImpl: public Operator
{

public:	
	static constexpr std::string_view kBackendName = BackendType::backend_name;

	OperatorImpl() = default;

	virtual ~OperatorImpl() {}
};

class OperatorShogunBackend
{
public:
	static constexpr std::string_view backend_name = "Shogun";
};

class OperatorNGraphBackend
{
public:
	static constexpr std::string_view backend_name = "NGraph";
};

}

#define BEGIN_OPERATOR_MANIFEST(DESCRIPTION)                             \
extern "C" Manifest shogunManifest()                            \
{                                                               \
	static Manifest manifest(DESCRIPTION,{                      \

/** Declares class to be exported.
 * Always use this macro between @ref BEGIN_MANIFEST and
 * @ref END_MANIFEST
 */
#define EXPORT_OPERATOR(CLASSNAME, ENGINE_TYPE, IDENTIFIER)                           \
	std::make_pair(std::string(IDENTIFIER).append(CLASSNAME::kBackendName), make_any(                                        \
		MetaClass<OperatorImpl<ENGINE_TYPE>>(make_any<std::shared_ptr<OperatorImpl<ENGINE_TYPE>>>(    \
				[]() -> std::shared_ptr<OperatorImpl<ENGINE_TYPE>>                         \
				{                                                               \
					return std::make_shared<CLASSNAME>();                       \
				}                                                               \
				)))),                                                           \
	std::make_pair(IDENTIFIER, make_any(                                        \
		MetaClass<OperatorImpl<ENGINE_TYPE>>(make_any<std::shared_ptr<OperatorImpl<ENGINE_TYPE>>>(    \
				[]() -> std::shared_ptr<OperatorImpl<ENGINE_TYPE>>                         \
				{                                                               \
					return std::make_shared<CLASSNAME>();                       \
				}                                                               \
				)))),                                                           \

/** Ends manifest declaration.
 * Always use this macro after @ref BEGIN_MANIFEST
 */
#define END_OPERATOR_MANIFEST()                                          \
		});                                                     \
	return manifest;                                            \
}

#endif