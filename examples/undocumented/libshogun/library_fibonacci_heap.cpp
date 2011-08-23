#include <shogun/lib/FibonacciHeap.h>
#include <stdio.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();
	double v[8] = {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7};
	int k[8] = {0,1,2,3,4,5,6,7};

	CFibonacciHeap* heap = new CFibonacciHeap(8);

	for (int i=0; i<8; i++)
		heap->insert(k[i],v[i]);
	
	int k_extract;
	double v_extract;
	for (int i=0; i<8; i++)
	{
		k_extract = heap->extract_min(v_extract);
		if (v[k_extract]!=v_extract)
		{
			printf("Fibonacci heap goes wrong.\n");
		}
	}
	
	exit_shogun();
	return 0;
}

