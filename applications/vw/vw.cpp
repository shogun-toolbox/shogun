#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>

#include <io/streaming/StreamingVwFile.h>
#include <io/streaming/StreamingVwCacheFile.h>
#include <features/streaming/StreamingVwFeatures.h>
#include <classifier/vw/VowpalWabbit.h>

using namespace shogun;

class Args_t
{
public:
	Args_t()
		{
			adaptive = false;
			exact_adaptive_norm = false;
			use_cache_input = false;
			create_cache = false;
			input_file_name = NULL;
			regressor_input_file_name = NULL;
			regressor_output_file_name = NULL;
			predictions_output_file_name = NULL;
		}

public:
	bool adaptive;
	bool exact_adaptive_norm;
	bool use_cache_input;
	bool create_cache;
	char* input_file_name;
	char* regressor_input_file_name;
	char* regressor_output_file_name;
	char* predictions_output_file_name;
} Args;

static const struct option longOpts[] = {
	{ "data", required_argument, NULL, 'd' },
	{ "adaptive", no_argument, NULL, 'a' },
	{ "exact_adaptive_norm", no_argument, NULL, 'e' },
	{ "use_cache", no_argument, NULL, 'c' },
	{ "create_cache", no_argument, NULL, 'C' },
	{ "predictions", required_argument, NULL, 'p' },
	{ "help", no_argument, NULL, 'h' }
};

static const char *optString = "d:aecCp:h";

void display_usage()
{
	printf("vw - Run Vowpal Wabbit.\n\n");
	printf("Supported arguments are:\n");
	printf("-d <file> \t-\tName of input file.\n");
	printf("-a\t-\tEnable adaptive learning.\n");
	printf("-e\t-\tUse exact norm during adaptive learning.\n");
	printf("-c\t-\tTry to use a cache file for input.\n");
	printf("-C\t-\tCreate a cache file from data.\n");
	printf("-p <file> \t-\tFile to write predictions to.\n");
	printf("-h\t-\tDisplay this information.\n");

	exit(1);
}

void parse_options(int argc, char** argv)
{
	int opt = 0;
	int longIndex;

	opt = getopt_long(argc, argv, optString, longOpts, &longIndex);
	while (opt != -1)
	{
		switch (opt)
		{
		case 'd':
			Args.input_file_name = optarg;
			printf("Input file is: %s.\n", Args.input_file_name);
			break;

		case 'a':
			Args.adaptive = true;
			printf("Using adaptive learning.\n");
			break;

		case 'e':
			Args.adaptive = true;
			Args.exact_adaptive_norm = true;
			printf("Using exact adaptive norm.\n");
			break;

		case 'c':
			Args.use_cache_input = true;
			printf("Treating input as a cache file.\n");
			break;

		case 'C':
			Args.create_cache = true;
			printf("Will create a cache file from the input.\n");
			break;

		case 'p':
			Args.predictions_output_file_name = optarg;
			printf("Predictions will be saved to: %s.\n", Args.predictions_output_file_name);
			break;

		case 'h':
			display_usage();
			break;

		default:
			break;
		}

		opt = getopt_long(argc, argv, optString, longOpts, &longIndex);
	}

	if (! Args.input_file_name)
	{
		printf("Data file must be specified! (use -d <file>)\n");
		exit(1);
	}

	if (Args.create_cache && Args.use_cache_input)
	{
		printf("Creating cache not supported while reading from cache input!\n");
		exit(1);
	}
}

void display_stats(CVowpalWabbit* vw)
{
	CVwEnvironment* env = vw->get_env();
	SG_REF(env);

	double weighted_labeled_examples = env->weighted_examples - env->weighted_unlabeled_examples;
	double best_constant = (env->weighted_labels - env->initial_t) / weighted_labeled_examples;
	double constant_loss = (best_constant*(1.0 - best_constant)*(1.0 - best_constant) + (1.0 - best_constant)*best_constant*best_constant);

	printf("\nFinished run.\n");
	printf("Number of examples = %lld.\n", env->example_number);
	printf("Weighted example sum = %f.\n", env->weighted_examples);
	printf("Weighted label sum = %f.\n", env->weighted_labels);
	printf("Average loss = %f.\n", env->sum_loss / env->weighted_examples);
	printf("Best constant = %f.\n", best_constant);

	if (env->min_label == 0. && env->max_label == 1. && best_constant < 1. && best_constant > 0.)
		printf("Best constant's loss = %f.\n", constant_loss);

	printf("Total feature number = %ld.\n", (long int) env->total_features);

	SG_UNREF(env);
}

int main(int argc, char** argv)
{
	parse_options(argc, argv);

	init_shogun_with_defaults();

	CStreamingVwFile* vw_file = NULL;
	CStreamingVwCacheFile* vw_cache_file = NULL;
	CStreamingVwFeatures* features = NULL;

	if (Args.use_cache_input)
	{
		vw_cache_file = new CStreamingVwCacheFile(Args.input_file_name);
		SG_REF(vw_cache_file);
		features = new CStreamingVwFeatures(vw_cache_file, true, 1024);
		SG_REF(features);
	}
	else
	{
		vw_file = new CStreamingVwFile(Args.input_file_name);
		SG_REF(vw_file);
		features = new CStreamingVwFeatures(vw_file, true, 1024);
		SG_REF(features);
	}

	CVowpalWabbit* vw = new CVowpalWabbit(features);

	if (Args.adaptive)
		vw->set_adaptive(true);

	if (Args.exact_adaptive_norm)
		vw->set_exact_adaptive_norm(true);

	if (Args.create_cache)
		vw_file->set_write_to_cache(true);

	if (Args.predictions_output_file_name)
		vw->set_prediction_out(Args.predictions_output_file_name);

	SG_REF(vw);
	vw->train();

	SG_REF(vw);
	display_stats(vw);

	if (Args.use_cache_input)
	{
		SG_UNREF(vw_cache_file);
	}
	else
	{
		SG_UNREF(vw_file);
	}

	SG_UNREF(features);
	SG_UNREF(vw);

	exit_shogun();

	return 0;
}
