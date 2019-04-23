#include <gmock/gmock.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{

	class MockTransformer : public Transformer
	{
	public:
		MOCK_METHOD1(fit, void(std::shared_ptr<Features>));

		MOCK_METHOD2(fit, void(std::shared_ptr<Features>, std::shared_ptr<Labels>));

		MOCK_METHOD2(transform, std::shared_ptr<Features>(std::shared_ptr<Features>, bool));

		MOCK_CONST_METHOD0(train_require_labels, bool());

		virtual const char* get_name() const
		{
			return "MockTransformer";
		}
	};

} // namespace shogun
