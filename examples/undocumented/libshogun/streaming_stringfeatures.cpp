#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/features/StreamingStringFeatures.h>

using namespace shogun;

void display_vector(const SGString<char> &vec)
{
	printf("\nNew Vector\n------------------\n");
	printf("Length=%d.\n", vec.slen);
	for (int32_t i=0; i<vec.slen; i++)
	{
		printf("%c", vec.string[i]);
	}
	printf("\n");
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	CStreamingAsciiFile* file = new CStreamingAsciiFile("../data/fm_train_dna.dat");

	CStreamingStringFeatures<char>* feat = new CStreamingStringFeatures<char>(file, false, 1024);
	feat->use_alphabet(DNA);

	feat->start_parser();
	while (feat->get_next_example())
	{
		SGString<char> vec = feat->get_vector();
		display_vector(vec);
		feat->release_example();
	}
	feat->end_parser();

	CAlphabet* alpha = feat->get_alphabet();

	printf("\nThe histogram is:\n");
	alpha->print_histogram();

	SG_UNREF(alpha);
	SG_UNREF(feat);
	SG_UNREF(file);

	exit_shogun();
	return 0;
}
