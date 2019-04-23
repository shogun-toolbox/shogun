#ifndef _MOCKCMACHINE_H_
#define _MOCKCMACHINE_H_

#include <gmock/gmock.h>
#include <shogun/machine/Machine.h>

namespace shogun {

	class MockMachine : public Machine {
		public:
			MOCK_METHOD1(apply, std::shared_ptr<Labels>(std::shared_ptr<Features>));
			MOCK_METHOD1(train_machine, bool(std::shared_ptr<Features>));
			MOCK_CONST_METHOD1(clone, std::shared_ptr<SGObject>(ParameterProperties));

			virtual const char* get_name() const { return "MockMachine"; }
	};

}  // namespace shogun
#endif /* _MOCKCMACHINE_H_ */
