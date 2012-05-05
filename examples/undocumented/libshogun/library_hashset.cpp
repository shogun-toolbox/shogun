#include <shogun/lib/Set.h>
#include <stdio.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();
	double v[8] = {0.0,0.0,0.1,0.1,0.2,0.2,0.3,0.3};

	CSet<double>* set = new CSet<double>(2, 8);

	for (int i=0; i<8; i++)
		set->add(v[i]);

	set->remove(0.1);
	set->add(0.4);

	exit_shogun();

	delete set;
	return 0;
}

