#ifndef ENUMS
#define ENUMS

namespace shogun
{
	namespace deeplearning
	{
		namespace enums
		{
			enum class FuncType
			{
				SIGM,
				TANH,
				LINEAR,
				RECTIFIED,
				SOFTMAX
			};

			enum class TaskType
			{
				UNRECOGNIZED,
				TRAIN_ALL,
				TEST_ALL,
				TRAIN_TEST_ALL,
				TRAIN_TEST_EPOCHS
			};
		}
	}
}

#endif
