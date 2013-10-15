#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGString.h>
#include <shogun/io/File.h>

namespace shogun
{

template <class T>
SGStringList<T>::SGStringList() : SGReferencedData()
{
	init_data();
}

template <class T>
SGStringList<T>::SGStringList(SGString<T>* s, index_t num_s, index_t max_length,
		bool ref_counting) :
	SGReferencedData(ref_counting), num_strings(num_s),
	max_string_length(max_length), strings(s)
{
}

template <class T>
SGStringList<T>::SGStringList(index_t num_s, index_t max_length, bool ref_counting) :
	SGReferencedData(ref_counting),
	num_strings(num_s), max_string_length(max_length)
{
	strings=SG_MALLOC(SGString<T>, num_strings);
}

template <class T>
SGStringList<T>::SGStringList(const SGStringList &orig) :
	SGReferencedData(orig)
{
	copy_data(orig);
}

template <class T>
SGStringList<T>::~SGStringList()
{
	unref();
}

template<class T> void SGStringList<T>::load(CFile* loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	loader->get_string_list(strings, num_strings, max_string_length);
	SG_RESET_LOCALE;
}

template<class T> void SGStringList<T>::save(CFile* saver)
{
	ASSERT(saver)

	SG_SET_LOCALE_C;
	saver->set_string_list(strings, num_strings);
	SG_RESET_LOCALE;
}


template <class T>
void SGStringList<T>::copy_data(const SGReferencedData &orig)
{
	strings = ((SGStringList*)(&orig))->strings;
	num_strings = ((SGStringList*)(&orig))->num_strings;
	max_string_length = ((SGStringList*)(&orig))->max_string_length;
}

template <class T>
void SGStringList<T>::init_data()
{
	strings = NULL;
	num_strings = 0;
	max_string_length = 0;
}

template <class T>
void SGStringList<T>::free_data()
{
	SG_FREE(strings);

	strings = NULL;
	num_strings = 0;
	max_string_length = 0;
}

template class SGStringList<bool>;
template class SGStringList<char>;
template class SGStringList<int8_t>;
template class SGStringList<uint8_t>;
template class SGStringList<int16_t>;
template class SGStringList<uint16_t>;
template class SGStringList<int32_t>;
template class SGStringList<uint32_t>;
template class SGStringList<int64_t>;
template class SGStringList<uint64_t>;
template class SGStringList<float32_t>;
template class SGStringList<float64_t>;
template class SGStringList<floatmax_t>;
}
