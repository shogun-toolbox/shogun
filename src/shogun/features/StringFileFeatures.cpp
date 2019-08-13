#include <shogun/features/StringFileFeatures.h>

namespace shogun
{

template <class ST> CStringFileFeatures<ST>::CStringFileFeatures() : CStringFeatures<ST>(), file(NULL)
{
}

template <class ST> CStringFileFeatures<ST>::CStringFileFeatures(const char* fname, EAlphabet alpha)
: CStringFeatures<ST>(alpha)
{
	file = new CMemoryMappedFile<ST>(fname);
	fetch_meta_info_from_file();
}

template <class ST> CStringFileFeatures<ST>::~CStringFileFeatures()
{
	SG_UNREF(file);
	CStringFileFeatures<ST>::cleanup();
}

template <class ST> ST* CStringFileFeatures<ST>::get_line(uint64_t& len, uint64_t& offs, int32_t& line_nr, uint64_t file_length)
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
			if (!CStringFeatures<ST>::alphabet->is_valid((uint8_t) c))
			{
				CStringFileFeatures<ST>::cleanup();
				SG_CLASS_ERROR(CStringFeatures<ST>, "Invalid character (%c) in line %d\n", c, line_nr)
			}
		}
	}

	len=0;
	offs=file_length;
	return NULL;
}

template <class ST> void CStringFileFeatures<ST>::cleanup()
{
	CStringFeatures<ST>::features.clear();
	CStringFeatures<ST>::symbol_mask_table = SGVector<ST>();

	/* start with a fresh alphabet, but instead of emptying the histogram
	 * create a new object (to leave the alphabet object alone if it is used
	 * by others)
	 */
	CAlphabet* alpha=new CAlphabet(CStringFeatures<ST>::alphabet->get_alphabet());
	SG_UNREF(CStringFeatures<ST>::alphabet);
	CStringFeatures<ST>::alphabet=alpha;
	SG_REF(CStringFeatures<ST>::alphabet);
}

template <class ST> void CStringFileFeatures<ST>::cleanup_feature_vector(int32_t num)
{
	SG_CLASS_ERROR(CStringFeatures<ST>, "Cleaning single feature vector not"
			"supported by StringFileFeatures\n")
}

template <class ST> void CStringFileFeatures<ST>::fetch_meta_info_from_file(int32_t granularity)
{
	CStringFileFeatures<ST>::cleanup();
	uint64_t file_size=file->get_size();
	ASSERT(granularity>=1)
	ASSERT(CStringFeatures<ST>::alphabet)

	int64_t buffer_size=granularity;
	CStringFeatures<ST>::features.clear();
	CStringFeatures<ST>::features.resize(buffer_size);

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
				CStringFeatures<ST>::features.resize(buffer_size+granularity);
				buffer_size+=granularity;
			}

			CStringFeatures<ST>::features[num_vectors-1]=SGVector<ST>(line, len, false);
		}
		else
			break;
	}

	SG_CLASS_INFO(CStringFeatures<ST>, "number of strings:%d\n", CStringFeatures<ST>::get_num_vectors())
	SG_CLASS_INFO(CStringFeatures<ST>,"maximum string length:%d\n", CStringFeatures<ST>::get_max_vector_length())
	SG_CLASS_INFO(CStringFeatures<ST>,"max_value_in_histogram:%d\n", CStringFeatures<ST>::alphabet->get_max_value_in_histogram())
	SG_CLASS_INFO(CStringFeatures<ST>,"num_symbols_in_histogram:%d\n", CStringFeatures<ST>::alphabet->get_num_symbols_in_histogram())

	if (!CStringFeatures<ST>::alphabet->check_alphabet_size() || !CStringFeatures<ST>::alphabet->check_alphabet())
		CStringFileFeatures<ST>::cleanup();

	CStringFeatures<ST>::features.resize(num_vectors);
}

template class CStringFileFeatures<bool>;
template class CStringFileFeatures<char>;
template class CStringFileFeatures<int8_t>;
template class CStringFileFeatures<uint8_t>;
template class CStringFileFeatures<int16_t>;
template class CStringFileFeatures<uint16_t>;
template class CStringFileFeatures<int32_t>;
template class CStringFileFeatures<uint32_t>;
template class CStringFileFeatures<int64_t>;
template class CStringFileFeatures<uint64_t>;
template class CStringFileFeatures<float32_t>;
template class CStringFileFeatures<float64_t>;
template class CStringFileFeatures<floatmax_t>;
}
