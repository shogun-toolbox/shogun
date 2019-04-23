#include <shogun/lib/Set.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>

using namespace shogun;

#define SIZE 8

int main(int argc, char** argv)
{
	double v[SIZE] = {0.0,0.1,0.2,0.2,0.3,0.4,0.5,0.5};

	CSet<double>* set = new CSet<double>(SIZE/2, SIZE/2);

	for (int i=0; i<SIZE; i++)
		set->add(v[i]);

	set->remove(0.2);

	//SG_SPRINT("Num of elements: %d\n", set->get_num_elements());
	for (int i=0; i<SIZE; i++)
	{
		if (set->contains(v[i]))
			;
			//SG_SPRINT("%lg contains in set with index %d\n", v[i], set->index_of(v[i]));
	}

	return 0;
}
