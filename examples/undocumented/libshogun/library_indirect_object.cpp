#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/IndirectObject.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>

#include <stdio.h>

using namespace shogun;

const int l=10;

int main(int argc, char** argv)
{
	init_shogun();

	// create array a
	int32_t* a=new int32_t[l];
	for (int i=0; i<l; i++)
		a[i]=l-i;

	// create array of indirect objects pointing to array a
	CIndirectObject<int32_t, int32_t**>::set_array(&a);
	CIndirectObject<int32_t, int32_t**>* x = new CIndirectObject<int32_t, int32_t**>[l];
	CIndirectObject<int32_t, int32_t**>::init_slice(x, l);


	printf("created array a and indirect object array x pointing to a.\n\n");
	for (int i=0; i<l; i++)
		printf("a[%d]=%d x[%d]=%d\n", i, a[i], i, int32_t(x[i]));

	//sort the array
	CMath::qsort(x, l);

	printf("\n\nvoila! sorted indirect object array x, keeping a const.\n\n");
	for (int i=0; i<l; i++)
		printf("a[%d]=%d x[%d]=%d\n", i, a[i], i, int32_t(x[i]));

	exit_shogun();

	return 0;
}
