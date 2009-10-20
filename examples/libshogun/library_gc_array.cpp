#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/GCArray.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/GaussianKernel.h>

#include <stdio.h>

using namespace shogun;

const int l=10;

int main(int argc, char** argv)
{
	init_shogun();

	// create array a
	CGCArray<CKernel*> kernels(l);

	for (int i=0; i<l; i++)
		kernels.set(new CGaussianKernel(10, 1.0), i);

	for (int i=0; i<l; i++)
		printf("kernels[%d]=%p\n", i, kernels.get(i));

	exit_shogun();

	return 0;
}
