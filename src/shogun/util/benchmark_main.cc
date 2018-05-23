#include <benchmark/benchmark.h>

#include <shogun/base/init.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	benchmark::Initialize(&argc, argv);
	benchmark::RunSpecifiedBenchmarks();

	exit_shogun();
	return 0;
}