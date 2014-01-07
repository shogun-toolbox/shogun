#include <gmock/gmock.h>
#include <labels/Labels.h>

namespace shogun {

	class MockCLabels : public CLabels {
		public:
			MOCK_METHOD1(ensure_valid, void(const char*));
			MOCK_CONST_METHOD0(get_num_labels, int32_t());
			MOCK_CONST_METHOD0(get_label_type, ELabelType());
			MOCK_METHOD0(get_values, SGVector<float64_t>());

			virtual const char* get_name() const { return "MockCLabels"; }
	};

}  // namespace shogun

