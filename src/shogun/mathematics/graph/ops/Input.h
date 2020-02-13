#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/Tensor.h>

namespace shogun {

	class Input: public Node
	{
	public:

		template <typename T>
		Input(const SGVector<T>& vec) : Node(std::make_shared<Tensor>(vec, Tensor::this_is_protected{0}))
		{
		}

		template <typename T>
		Input(const SGMatrix<T>& matrix) : Node(std::make_shared<Tensor>(matrix, Tensor::this_is_protected{0}))
		{
		}

		~Input();

	private:
		// noop in this case
		void evaluate() {}
		void allocate_tensor(const Shape& shape, element_type type) {}
	};


}