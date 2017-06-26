#include <shogun/lib/SGMatrix.h>
#include <memory>

using namespace shogun;

/** Fixture class that sets up data to be used in the kernel exponential family
 * unit tests. Result reference is at
 * https://gist.github.com/karlnapf/c0b24fc18d946cc315733ed679e249e8
 *
 */
class DataFixture
{
public:
	void SetUp() {
		N = 3;
		D = 2;
		ND = N * D;

		X_train_fixed = SGMatrix < float64_t > (D, N);
		X_train_fixed(0, 0) = 0;
		X_train_fixed(1, 0) = 1;
		X_train_fixed(0, 1) = 2;
		X_train_fixed(1, 1) = 4;
		X_train_fixed(0, 2) = 3;
		X_train_fixed(1, 2) = 6;

		N_test=2;
		X_test_fixed = SGMatrix < float64_t > (D, N_test);
		X_test_fixed(0, 0) = 0;
		X_test_fixed(1, 0) = 1;
		X_test_fixed(0, 1) = 1;
		X_test_fixed(1, 1) = 1;

		X_train_random = SGMatrix < float64_t > (D, N);
		for (auto i = 0; i < ND; i++)
			X_train_random.matrix[i] = CMath::randn_float();
	}

protected:
	index_t N;
	index_t N_test;
	index_t D;
	index_t ND;
	SGMatrix<float64_t> X_train_fixed;
	SGMatrix<float64_t> X_test_fixed;
	SGMatrix<float64_t> X_train_random;
};
