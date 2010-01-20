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

	uint32_t h=CHash::IncrementalMurmurHash2(array[0], 0xDEADBEAF);
	printf("inc_hash(0)=%0x\n", h);
	h=CHash::IncrementalMurmurHash2(array[1], h);
	printf("inc_hash(1)=%0x\n", h);
	h=CHash::IncrementalMurmurHash2(array[2], h);
	printf("inc_hash(2)=%0x\n", h);
	h=CHash::IncrementalMurmurHash2(array[3], h);
	printf("inc_hash(3)=%0x\n", h);
	exit_shogun();
	return 0;
}

