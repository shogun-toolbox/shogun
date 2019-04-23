/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Heiko Strathmann, Bjoern Esser,
 *          Sergey Lisitsyn, Viktor Gal
 */

#include <shogun/features/streaming/StreamingHashedDocDotFeatures.h>
#include <shogun/features/hashed/HashedDocDotFeatures.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

StreamingHashedDocDotFeatures::StreamingHashedDocDotFeatures(std::shared_ptr<StreamingFile> file,
	bool is_labelled, int32_t size,	std::shared_ptr<Tokenizer> tzer, int32_t bits)
: StreamingDotFeatures()
{
	init(file, is_labelled, size, tzer, bits, true, 1, 0);
}

StreamingHashedDocDotFeatures::StreamingHashedDocDotFeatures() : StreamingDotFeatures()
{
	init(NULL, false, 0, NULL, 0, false, 1, 0);
}

StreamingHashedDocDotFeatures::StreamingHashedDocDotFeatures(
	std::shared_ptr<StringFeatures<char>> dot_features, std::shared_ptr<Tokenizer> tzer, int32_t bits, float64_t* lab)
: StreamingDotFeatures()
{
	auto file =
		std::make_shared<StreamingFileFromStringFeatures<char>>(dot_features, lab);
	bool is_labelled = (lab != NULL);
	int32_t size=1024;

	init(file, is_labelled, size, tzer, bits, true, 1, 0);

	parser.set_free_vectors_on_destruct(false);
	seekable= true;
}
void StreamingHashedDocDotFeatures::init(std::shared_ptr<StreamingFile> file, bool is_labelled,
	int32_t size, std::shared_ptr<Tokenizer> tzer, int32_t bits, bool normalize, int32_t n_grams, int32_t skips)
{
	num_bits = bits;
	tokenizer = tzer;
	if (tokenizer)
	{

		converter = std::make_shared<HashedDocConverter>(tzer, bits, normalize, n_grams, skips);
	}
	else
		converter=NULL;

	SG_ADD(&num_bits, "num_bits", "Number of bits for hash");
	SG_ADD((std::shared_ptr<SGObject>* ) &tokenizer, "tokenizer", "The tokenizer used on the documents");
	SG_ADD((std::shared_ptr<SGObject>* ) &converter, "converter", "Converter");

	has_labels = is_labelled;
	if (file)
	{
		working_file = file;

		parser.init(file, is_labelled, size);
		seekable = false;
	}
	else
		working_file = NULL;

	set_read_functions();
	parser.set_free_vector_after_release(false);
}

StreamingHashedDocDotFeatures::~StreamingHashedDocDotFeatures()
{
	if (parser.is_running())
		parser.end_parser();



}

float32_t StreamingHashedDocDotFeatures::dot(std::shared_ptr<StreamingDotFeatures> df)
{
	ASSERT(df)
	ASSERT(df->get_name() == get_name())

	auto cdf = std::static_pointer_cast<StreamingHashedDocDotFeatures>(df);
	float32_t result = current_vector.sparse_dot(cdf->current_vector);
	return result;
}

float32_t StreamingHashedDocDotFeatures::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == Math::pow(2, num_bits))

	float32_t result = 0;
	for (index_t i=0; i<current_vector.num_feat_entries; i++)
	{
		result += vec2[current_vector.features[i].feat_index] *
					current_vector.features[i].entry;
	}
	return result;
}

void StreamingHashedDocDotFeatures::add_to_dense_vec(float32_t alpha, float32_t* vec2,
			int32_t vec2_len, bool abs_val)
{
	float32_t value = abs_val ? Math::abs(alpha) : alpha;

	for (index_t i=0; i<current_vector.num_feat_entries; i++)
		vec2[current_vector.features[i].feat_index] += value * current_vector.features[i].entry;
}

int32_t StreamingHashedDocDotFeatures::get_dim_feature_space() const
{
	return Math::pow(2, num_bits);
}

const char* StreamingHashedDocDotFeatures::get_name() const
{
	return "StreamingHashedDocDotFeatures";
}

EFeatureType StreamingHashedDocDotFeatures::get_feature_type() const
{
	return F_UINT;
}

EFeatureClass StreamingHashedDocDotFeatures::get_feature_class() const
{
	return C_STREAMING_SPARSE;
}

void StreamingHashedDocDotFeatures::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

void StreamingHashedDocDotFeatures::end_parser()
{
	parser.end_parser();
}

bool StreamingHashedDocDotFeatures::get_next_example()
{
	SGVector<char> tmp;
	if (parser.get_next_example(tmp.vector,
		tmp.vlen, current_label))
	{
		ASSERT(tmp.vector)
		ASSERT(tmp.vlen > 0)
		current_vector = converter->apply(tmp);
		return true;
	}
	return false;
}

void StreamingHashedDocDotFeatures::release_example()
{
	parser.finalize_example();
}

int32_t StreamingHashedDocDotFeatures::get_num_features()
{
	return (int32_t) Math::pow(2, num_bits);
}

float64_t StreamingHashedDocDotFeatures::get_label()
{
	return current_label;
}

int32_t StreamingHashedDocDotFeatures::get_num_vectors() const
{
	return 1;
}

void StreamingHashedDocDotFeatures::set_vector_reader()
{
	parser.set_read_vector(&StreamingFile::get_string);
}

void StreamingHashedDocDotFeatures::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&StreamingFile::get_string_and_label);
}

SGSparseVector<float64_t> StreamingHashedDocDotFeatures::get_vector()
{
	return current_vector;
}

void StreamingHashedDocDotFeatures::set_normalization(bool normalize)
{
	converter->set_normalization(normalize);
}

void StreamingHashedDocDotFeatures::set_k_skip_n_grams(int32_t k, int32_t n)
{
	converter->set_k_skip_n_grams(k, n);
}
