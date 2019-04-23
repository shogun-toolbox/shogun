#include <shogun/features/StringFileFeatures.h>

namespace shogun
{

template <class ST> StringFileFeatures<ST>::StringFileFeatures() : StringFeatures<ST>(), file(NULL)
{
}

template <class ST> StringFileFeatures<ST>::StringFileFeatures(const char* fname, EAlphabet alpha)
: StringFeatures<ST>(alpha)
{
	file = std::make_shared<MemoryMappedFile<ST>>(fname);
	fetch_meta_info_from_file();
}

template <class ST> StringFileFeatures<ST>::~StringFileFeatures()
{
	StringFileFeatures<ST>::cleanup();
}

template <class ST> ST* StringFileFeatures<ST>::get_line(uint64_t& len, uint64_t& offs, int32_t& line_nr, uint64_t file_length)
{
	ST* s = file->get_map();
	for (uint64_t i=offs; i<file_length; i++)
	{
		ST c=s[i];

		if ((char) c == '\n')
		{
			ST* line=&s[offs];
			len=i-offs;
			offs=i+1;
			line_nr++;
			return line;
		}
		else
		{
			if (!StringFeatures<ST>::alphabet->is_valid((uint8_t) c))
			{
				StringFileFeatures<ST>::cleanup();
				error("Invalid character ({}) in line {}", c, line_nr);
			}
		}
	}

	len=0;
	offs=file_length;
	return NULL;
}

template <class ST> void StringFileFeatures<ST>::cleanup()
{
	StringFeatures<ST>::features.clear();
	StringFeatures<ST>::symbol_mask_table = SGVector<ST>();

	/* start with a fresh alphabet, but instead of emptying the histogram
	* create a new object (to leave the alphabet object alone if it is used
	* by others)
	*/
	auto alpha=std::make_shared<Alphabet>(StringFeatures<ST>::alphabet->get_alphabet());
	StringFeatures<ST>::alphabet=alpha;
}

template <class ST> void StringFileFeatures<ST>::cleanup_feature_vector(int32_t num)
{
	error("Cleaning single feature vector not"
			"supported by StringFileFeatures");
}

template <class ST> void StringFileFeatures<ST>::fetch_meta_info_from_file(int32_t granularity)
{
	StringFileFeatures<ST>::cleanup();
	uint64_t file_size=file->get_size();
	ASSERT(granularity>=1)
	ASSERT(StringFeatures<ST>::alphabet)

	int64_t buffer_size=granularity;
	StringFeatures<ST>::features.clear();
	StringFeatures<ST>::features.resize(buffer_size);

	uint64_t offs=0;
	uint64_t len=0;
	int num_vectors=0;

	while (true)
	{
		ST* line=get_line(len, offs, num_vectors, file_size);

		if (line)
		{
			if (num_vectors > buffer_size)
			{
				StringFeatures<ST>::features.resize(buffer_size+granularity);
				buffer_size+=granularity;
			}

			StringFeatures<ST>::features[num_vectors-1]=SGVector<ST>(line, len, false);
		}
		else
			break;
	}

	io::info("number of strings:{}", StringFeatures<ST>::get_num_vectors());
	io::info("maximum string length:{}", StringFeatures<ST>::get_max_vector_length());
	io::info("max_value_in_histogram:{}", StringFeatures<ST>::alphabet->get_max_value_in_histogram());
	io::info("num_symbols_in_histogram:{}", StringFeatures<ST>::alphabet->get_num_symbols_in_histogram());

	if (!StringFeatures<ST>::alphabet->check_alphabet_size() || !StringFeatures<ST>::alphabet->check_alphabet())
		StringFileFeatures<ST>::cleanup();

	StringFeatures<ST>::features.resize(num_vectors);
}

template class StringFileFeatures<bool>;
template class StringFileFeatures<char>;
template class StringFileFeatures<int8_t>;
template class StringFileFeatures<uint8_t>;
template class StringFileFeatures<int16_t>;
template class StringFileFeatures<uint16_t>;
template class StringFileFeatures<int32_t>;
template class StringFileFeatures<uint32_t>;
template class StringFileFeatures<int64_t>;
template class StringFileFeatures<uint64_t>;
template class StringFileFeatures<float32_t>;
template class StringFileFeatures<float64_t>;
template class StringFileFeatures<floatmax_t>;
}
