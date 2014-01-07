#include <gmock/gmock.h>
#include <latent/LatentModel.h>

namespace shogun {

	class MockCLatentModel : public CLatentModel {
		public:
			MOCK_CONST_METHOD0(get_num_vectors, int32_t());
			MOCK_CONST_METHOD0(get_dim, int32_t());
			MOCK_METHOD0(get_psi_feature_vectors, CDotFeatures*());
			MOCK_METHOD2(infer_latent_variable, CData*(const SGVector<float64_t>& w, index_t idx));
			MOCK_METHOD1(argmax_h, void(const SGVector<float64_t>& w));
			MOCK_CONST_METHOD0(get_name, const char*());
	};

}  // namespace shogun

