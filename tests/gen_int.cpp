#include <stdio.h>

int main()
{
	int I;

	FILE* f=fopen("foo", "wb");
	for (int i=0; i<1000; i++)
	{
			I= (-1) * (i%2);
			fwrite(&I, sizeof(int), 1, f);
	}
	fclose(f);
	return 0;
}
