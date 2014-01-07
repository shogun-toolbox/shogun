#include <base/init.h>
#include <lib/common.h>
#include <lib/memory.h>
#include <lib/IndirectObject.h>
#include <mathematics/Math.h>
#include <base/SGObject.h>

#include <stdio.h>

using namespace shogun;

const int l=10;

int main(int argc, char** argv)
{
	init_shogun();

	// create array a
	int32_t* a=SG_MALLOC(int32_t, l);
	for (int i=0; i<l; i++)
		a[i]=l-i;
	typedef CIndirectObject<int32_t, int32_t**> INDIRECT;
	// create array of indirect objects pointing to array a
	INDIRECT::set_array(&a);
	INDIRECT* x = SG_MALLOC(INDIRECT, l);
	INDIRECT::init_slice(x, l);


	printf("created array a and indirect object array x pointing to a.\n\n");
	for (int i=0; i<l; i++)
		printf("a[%d]=%d x[%d]=%d\n", i, a[i], i, int32_t(x[i]));

	//sort the array
	CMath::qsort(x, l);

	printf("\n\nvoila! sorted indirect object array x, keeping a const.\n\n");
	for (int i=0; i<l; i++)
		printf("a[%d]=%d x[%d]=%d\n", i, a[i], i, int32_t(x[i]));

	SG_FREE(x);
	SG_FREE(a);
	exit_shogun();

	return 0;
}
