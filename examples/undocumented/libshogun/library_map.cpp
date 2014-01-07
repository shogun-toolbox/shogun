#include <lib/Map.h>
#include <io/SGIO.h>
#include <base/init.h>
#include <lib/common.h>

using namespace shogun;

#define SIZE 6

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message, &print_message, &print_message);
	const char* v[SIZE] = {"Russia", "England", "Germany", "USA", "France", "Spain"};

	CMap<int32_t, const char*>* map = new CMap<int32_t, const char*>(SIZE/2, SIZE/2);

	for (int i=0; i<SIZE; i++)
		map->add(i, v[i]);

	map->remove(0);

	//SG_SPRINT("Num of elements: %d\n", map->get_num_elements());
	for (int i=0; i<SIZE; i++)
	{
		if (map->contains(i))
			;
			//SG_SPRINT("key %d contains in map with index %d and data=%s\n",
			//	i, map->index_of(i), map->get_element(i));
	}

	SG_UNREF(map);
	exit_shogun();
	return 0;
}
