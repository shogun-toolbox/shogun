#include <benchmark/benchmark.h>

using namespace shogun;

int main(int argc, char** argv)
{
	benchmark::Initialize(&argc, argv);
	benchmark::RunSpecifiedBenchmarks();

	exit_shogun();
	return 0;
}