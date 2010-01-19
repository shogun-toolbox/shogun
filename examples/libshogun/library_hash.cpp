#include <shogun/lib/Hash.h>
#include <stdio.h>


using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();
	uint8_t array[4]={0,1,2,3};

	printf("hash(0)=%0x\n", CHash::MurmurHash2(&array[0], 1, 0xDEADBEAF));
	printf("hash(1)=%0x\n", CHash::MurmurHash2(&array[1], 1, 0xDEADBEAF));
	printf("hash(2)=%0x\n", CHash::MurmurHash2(&array[0], 2, 0xDEADBEAF));
	printf("hash(3)=%0x\n", CHash::MurmurHash2(&array[0], 4, 0xDEADBEAF));

	exit_shogun();
	return 0;
}

