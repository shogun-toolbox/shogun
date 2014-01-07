#include <base/init.h>
#include <io/LineReader.h>
#include <lib/DelimiterTokenizer.h>
#include <lib/SGVector.h>

#include <cstdio>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	FILE* fin=fopen("io_linereader.cpp", "r");

	CDelimiterTokenizer* tokenizer=new CDelimiterTokenizer();
	tokenizer->delimiters['\n']=1;
	SG_REF(tokenizer);

	CLineReader* reader=new CLineReader(fin, tokenizer);

	int lines_count=0;
	SGVector<char> tmp_string;
	while (reader->has_next())
	{
		tmp_string=reader->read_line();
		SG_SPRINT("%d %d ", lines_count, tmp_string.vlen);
		for (int i=0; i<tmp_string.vlen; i++)
			SG_SPRINT("%c", tmp_string.vector[i]);
		SG_SPRINT("\n");
		lines_count++;
	}
	SG_SPRINT("total lines: %d\n", lines_count);

	tmp_string=SGVector<char>();
	SG_UNREF(reader);
	SG_UNREF(tokenizer);

	fclose(fin);

	exit_shogun();
	return 0;
}
