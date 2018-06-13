#include <gmock/gmock.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

	class MockCTransformer : public CTransformer
	{
	public:
		MOCK_METHOD1(fit, void(CFeatures*));

		MOCK_METHOD2(fit, void(CFeatures*, CLabels*));

		MOCK_METHOD2(transform, CFeatures*(CFeatures*, bool));

		MOCK_CONST_METHOD0(train_require_labels, bool());

		virtual const char* get_name() const
		{
			return "MockCTransformer";
		}
	};

} // namespace shogun
