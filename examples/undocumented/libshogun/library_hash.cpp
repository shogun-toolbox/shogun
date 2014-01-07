#include <lib/Hash.h>
#include <stdio.h>


using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();
	uint8_t array[4]={0,1,2,3};

	printf("hash(0)=%0x\n", CHash::MurmurHash3(&array[0], 1, 0xDEADBEAF));
	printf("hash(1)=%0x\n", CHash::MurmurHash3(&array[1], 1, 0xDEADBEAF));
	printf("hash(2)=%0x\n", CHash::MurmurHash3(&array[0], 2, 0xDEADBEAF));
	printf("hash(3)=%0x\n", CHash::MurmurHash3(&array[0], 4, 0xDEADBEAF));

	uint32_t h = 0xDEADBEAF;
	uint32_t carry = 0;
	CHash::IncrementalMurmurHash3(&h, &carry, &array[0], 1);
	printf("inc_hash(0)=%0x\n", h);
	CHash::IncrementalMurmurHash3(&h, &carry, &array[1], 1);
	printf("inc_hash(1)=%0x\n", h);
	CHash::IncrementalMurmurHash3(&h, &carry, &array[2], 1);
	printf("inc_hash(2)=%0x\n", h);
	CHash::IncrementalMurmurHash3(&h, &carry, &array[3], 1);
	printf("inc_hash(3)=%0x\n", h);
	h = CHash::FinalizeIncrementalMurmurHash3(h, carry, 4);
        printf("Final inc_hash(3)=%0x\n", h);
	exit_shogun();
	return 0;
}

