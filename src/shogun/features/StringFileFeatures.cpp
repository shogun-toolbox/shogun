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
				SG_CLASS_ERROR(StringFeatures<ST>, "Invalid character (%c) in line %d\n", c, line_nr)
			}
		}
	}

	len=0;
	offs=file_length;
	return NULL;
}

template <class ST> void StringFileFeatures<ST>::cleanup()
{
	StringFeatures<ST>::num_vectors=0;
	SG_FREE(StringFeatures<ST>::features);
	SG_FREE(StringFeatures<ST>::symbol_mask_table);
	StringFeatures<ST>::features=NULL;
	StringFeatures<ST>::symbol_mask_table=NULL;

	/* start with a fresh alphabet, but instead of emptying the histogram
	* create a new object (to leave the alphabet object alone if it is used
	* by others)
	*/
	auto alpha=std::make_shared<Alphabet>(StringFeatures<ST>::alphabet->get_alphabet());
	StringFeatures<ST>::alphabet=alpha;
}

template <class ST> void StringFileFeatures<ST>::cleanup_feature_vector(int32_t num)
{
	SG_CLASS_ERROR(StringFeatures<ST>, "Cleaning single feature vector not"
		"supported by StringFileFeatures\n")
}

template <class ST> void StringFileFeatures<ST>::fetch_meta_info_from_file(int32_t granularity)
{
	StringFileFeatures<ST>::cleanup();
	uint64_t file_size=file->get_size();
	ASSERT(granularity>=1)
	ASSERT(StringFeatures<ST>::alphabet)

	int64_t buffer_size=granularity;
	StringFeatures<ST>::features=SG_MALLOC(SGString<ST>, buffer_size);

	uint64_t offs=0;
	uint64_t len=0;
	StringFeatures<ST>::max_string_length=0;
	StringFeatures<ST>::num_vectors=0;

	while (true)
	{
		ST* line=get_line(len, offs, StringFeatures<ST>::num_vectors, file_size);

		if (line)
		{
			if (StringFeatures<ST>::num_vectors > buffer_size)
			{
				StringFeatures<ST>::features = SG_REALLOC(SGString<ST>, StringFeatures<ST>::features, buffer_size, buffer_size+granularity);
				buffer_size+=granularity;
			}

			StringFeatures<ST>::features[StringFeatures<ST>::num_vectors-1].string=line;
			StringFeatures<ST>::features[StringFeatures<ST>::num_vectors-1].slen=len;
			StringFeatures<ST>::max_string_length=Math::max(StringFeatures<ST>::max_string_length, (int32_t) len);
		}
		else
			break;
	}

	SG_CLASS_INFO(StringFeatures<ST>, "number of strings:%d\n", StringFeatures<ST>::num_vectors)
	SG_CLASS_INFO(StringFeatures<ST>,"maximum string length:%d\n", StringFeatures<ST>::max_string_length)
	SG_CLASS_INFO(StringFeatures<ST>,"max_value_in_histogram:%d\n", StringFeatures<ST>::alphabet->get_max_value_in_histogram())
	SG_CLASS_INFO(StringFeatures<ST>,"num_symbols_in_histogram:%d\n", StringFeatures<ST>::alphabet->get_num_symbols_in_histogram())

	if (!StringFeatures<ST>::alphabet->check_alphabet_size() || !StringFeatures<ST>::alphabet->check_alphabet())
		StringFileFeatures<ST>::cleanup();

	StringFeatures<ST>::features=SG_REALLOC(SGString<ST>, StringFeatures<ST>::features, buffer_size, StringFeatures<ST>::num_vectors);
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
