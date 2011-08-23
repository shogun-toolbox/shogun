#include <shogun/lib/HashSet.h>
#include <stdio.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();
	double v[8] = {0.0,0.0,0.1,0.1,0.2,0.2,0.3,0.3};

	CHashSet* set = new CHashSet(8);

	for (int i=0; i<8; i++)
		set->insert_key(i,v[i]);

	delete set;
	exit_shogun();
	return 0;
}

