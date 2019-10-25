#include <gmock/gmock.h>
#include <shogun/features/Features.h>

namespace shogun {

	class MockFeatures : public Features {
		public:
			MOCK_CONST_METHOD0(duplicate, std::shared_ptr<Features>());
			MOCK_CONST_METHOD0(get_feature_type, EFeatureType());
			MOCK_CONST_METHOD0(get_feature_class, EFeatureClass());
			MOCK_CONST_METHOD0(get_num_vectors, int32_t());

			virtual const char* get_name() const { return "MockFeatures"; }
	};

}  // namespace shogun

