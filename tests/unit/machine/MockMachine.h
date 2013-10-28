#ifndef _MOCKCMACHINE_H_
#define _MOCKCMACHINE_H_

#include <gmock/gmock.h>
#include <shogun/machine/Machine.h>

namespace shogun {

	class MockCMachine : public CMachine {
		public:
			MOCK_METHOD1(apply, CLabels*(CFeatures*));
			MOCK_METHOD1(train_locked, bool(SGVector<index_t>));
			MOCK_METHOD1(train_machine, bool(CFeatures*));
			MOCK_METHOD0(store_model_features, void());
			MOCK_METHOD0(clone, CSGObject*());

			virtual const char* get_name() const { return "MockCMachine"; }
	};

}  // namespace shogun
#endif /* _MOCKCMACHINE_H_ */
