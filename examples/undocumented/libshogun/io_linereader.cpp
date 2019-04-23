#include <shogun/base/init.h>
#include <shogun/io/LineReader.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>

#include <cstdio>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	FILE* fin=fopen("../data/label_train_multiclass_digits.dat", "r");

	auto tokenizer=std::make_shared<DelimiterTokenizer>();
	tokenizer->delimiters['\n']=1;

	auto reader=std::make_shared<LineReader>(fin, tokenizer);

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

	fclose(fin);

	exit_shogun();
	return 0;
}
