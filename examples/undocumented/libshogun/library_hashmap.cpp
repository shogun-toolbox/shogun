#include <shogun/lib/HashMap.h>
#include <stdio.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();
	float64_t v[8] = {0.0,0.0,0.1,0.1,0.2,0.2,0.3,0.3};
	float64_t temp;

	CHashMap<float64_t>* set = new CHashMap<float64_t>(8);

	for (int i=0; i< 8 * 10; i++)
	{
		set->insert_key(i, v[i % 8]);
		set->search_key(i, temp);

		printf("%g\n", temp);
	}

	delete set;
	exit_shogun();
	return 0;
}

