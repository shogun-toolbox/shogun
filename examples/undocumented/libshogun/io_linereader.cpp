#include <shogun/base/init.h>

#include <shogun/io/LineReader.h>
#include <shogun/lib/SGVector.h>

#include <cstdio>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	FILE* fin=fopen("io_linereader.cpp", "r");
	CLineReader* reader=new CLineReader(fin);

	int lines_count=0;
	SGVector<char> tmp_string;
	while (reader->has_next_line())
	{
		tmp_string=reader->get_next_line();
		SG_SPRINT("%d %d ", lines_count, tmp_string.vlen);
		for (int i=0; i<tmp_string.vlen; i++)
			SG_SPRINT("%c", tmp_string.vector[i]);
		SG_SPRINT("\n");
		lines_count++;
	}
	SG_SPRINT("total lines: %d\n", lines_count);

	delete reader;
	fclose(fin);

	exit_shogun();
	return 0;
}
