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

template <typename BackendType, typename OperatorType>
IGNORE_IN_CLASSLIST class OperatorImpl: public Operator
{

public:
	using operator_type = OperatorType;

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

#define REGISTER_OP_FACTORY(opr, OP) \
  REGISTER_OP_UNIQ_HELPER(__COUNTER__, opr, OP)
#define REGISTER_OP_UNIQ_HELPER(ctr, opr, OP) \
  REGISTER_OP_UNIQ(ctr, opr, OP)
#define REGISTER_OP_UNIQ(ctr, opr, OP)   \
  static auto register_opf##ctr SG_ATTRIBUTE_UNUSED =    \
		  opr.emplace(								\
			std::type_index(typeid(OP::operator_type)), []() { return std::make_shared<OP>(); })
#define REGISTER_OP(OP) 						\
	REGISTER_OP_FACTORY(OperatorRegistry(), OP)

}


#endif