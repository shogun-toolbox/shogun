/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#include <io/SGIO.h>
#include <lib/Time.h>
#include <lib/ShogunException.h>
#include <mathematics/Math.h>
#include <lib/DynInt.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void print_warning(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void print_error(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void gen_ints(uint256_t* &a, uint32_t* &b, uint32_t len)
{
	a=SG_MALLOC(uint256_t, len);
	b=SG_MALLOC(uint32_t, len);

	CMath::init_random(17);

	for (uint32_t i=0; i<len; i++)
	{
		uint64_t r[4]={(uint64_t) CMath::random() << 32 | CMath::random(),
			(uint64_t) CMath::random() << 32 | CMath::random(),
			(uint64_t) CMath::random() << 32 | CMath::random(),
			(uint64_t) CMath::random() << 32 | CMath::random()};

		a[len-i-1]=r;
		b[len-i-1]=i;
	}
}

const int LEN = 5*1024;

int main()
{
	init_shogun(&print_message, &print_warning,
			&print_error);
	try
	{
		uint256_t* a;
		uint32_t* b;
		CTime t;
		t.io->set_loglevel(MSG_DEBUG);

		SG_SPRINT("gen data..");
		t.start();
		gen_ints(a,b, LEN);
		t.cur_time_diff(true);

		SG_SPRINT("qsort..");
		t.start();
		CMath::qsort_index(a, b, LEN);
		t.cur_time_diff(true);

		SG_SPRINT("\n\n");
		for (uint32_t i=0; i<10; i++)
		{
			SG_SPRINT("a[%d]=", i);
			a[i].print_hex();
			SG_SPRINT("\n");
		}

		SG_SPRINT("\n\n");

		uint64_t val1[4]={1,2,3,4};
		uint64_t val2[4]={5,6,7,8};
		a[0]=val1;
		a[1]=val2;
		a[2]=a[0];
		CMath::swap(a[0],a[1]);

		printf("a[0]==a[1] %d\n", (int) (a[0] == a[1]));
		printf("a[0]<a[1] %d\n", (int) (a[0] < a[1]));
		printf("a[0]<=a[1] %d\n", (int) (a[0] <= a[1]));
		printf("a[0]>a[1] %d\n", (int) (a[0] > a[1]));
		printf("a[0]>=a[1] %d\n", (int) (a[0] >= a[1]));

		printf("a[0]==a[0] %d\n", (int) (a[0] == a[0]));
		printf("a[0]<a[0] %d\n", (int) (a[0] < a[0]));
		printf("a[0]<=a[0] %d\n", (int) (a[0] <= a[0]));
		printf("a[0]>a[0] %d\n", (int) (a[0] > a[0]));
		printf("a[0]>=a[0] %d\n", (int) (a[0] >= a[0]));

		SG_SPRINT("\n\n");
		for (uint32_t i=0; i<10 ; i++)
		{
			SG_SPRINT("a[%d]=", i);
			a[i].print_hex();
			printf("\n");
		}

		SG_FREE(a);
		SG_FREE(b);
	}
	catch(ShogunException & sh)
	{
		SG_SPRINT("%s",sh.get_exception_string());
	}

	exit_shogun();

	return 0;
}
