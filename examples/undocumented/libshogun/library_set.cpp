#include <shogun/lib/Set.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>

using namespace shogun;

#define SIZE 8

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message, &print_message, &print_message);
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

	SG_UNREF(set);
	exit_shogun();
	return 0;
}
