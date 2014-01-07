#include <base/init.h>
#include <lib/CircularBuffer.h>
#include <lib/DelimiterTokenizer.h>
#include <lib/SGVector.h>

#include <cstdio>
#include <cstring>

using namespace shogun;

const int max_line_length = 256;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	SGVector<char> test_string(const_cast<char* >("all your bayes are belong to us! "), 33, false);

	CCircularBuffer* buffer=new CCircularBuffer(max_line_length);

	CDelimiterTokenizer* tokenizer=new CDelimiterTokenizer();
	tokenizer->delimiters[' ']=1;
	SG_REF(tokenizer);

	buffer->set_tokenizer(tokenizer);

	SGVector<char> tmp_string;
	buffer->push(test_string);

	int num_read;
	index_t start;
	while ((num_read=buffer->next_token_idx(start))>0)
	{
		buffer->skip_characters(start);
		tmp_string=buffer->pop(num_read);
		buffer->skip_characters(1);
		for (int i=0; i<tmp_string.vlen; i++)
			SG_SPRINT("%c", tmp_string.vector[i]);
		SG_SPRINT("\n");
	}

	SG_UNREF(buffer);
	SG_UNREF(tokenizer);

	exit_shogun();
	return 0;
}
