#include <stdio.h>

int main()
{
	double d=0.0;

	FILE* f=fopen("foo", "wb");
	for (int i=0; i<1000; i++)
	{
		for (int j=0; j<120; j++)
		{
			d=i+j/1000.0;
			fwrite(&d, sizeof(double), 1, f);
		}
	}
	fclose(f);
	return 0;
}
