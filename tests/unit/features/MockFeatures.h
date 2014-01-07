#include <gmock/gmock.h>
#include <features/Features.h>

namespace shogun {

	class MockCFeatures : public CFeatures {
		public:
			MOCK_CONST_METHOD0(duplicate, CFeatures*());
			MOCK_CONST_METHOD0(get_feature_type, EFeatureType());
			MOCK_CONST_METHOD0(get_feature_class, EFeatureClass());
			MOCK_CONST_METHOD0(get_num_vectors, int32_t());

			virtual const char* get_name() const { return "MockCFeatures"; }
	};

}  // namespace shogun

