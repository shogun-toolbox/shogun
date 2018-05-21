#include <shogun/preprocessor/DecompressString.h>

namespace shogun
{

/** default constructor  */
template <class ST>
CDecompressString<ST>::CDecompressString() : CStringPreprocessor<ST>()
{
	compressor=NULL;
	CSGObject::set_generic<ST>();
}

template <class ST>
CDecompressString<ST>::CDecompressString(E_COMPRESSION_TYPE ct) : CStringPreprocessor<ST>()
{
	compressor=new CCompressor(ct);
	CSGObject::set_generic<ST>();
}

template <class ST>
CDecompressString<ST>::~CDecompressString()
{
	delete compressor;
}

template <class ST>
void CDecompressString<ST>::cleanup()
{
}

template <class ST>
bool CDecompressString<ST>::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

template <class ST>
bool CDecompressString<ST>::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

template <class ST>
void CDecompressString<ST>::apply_to_string_list(SGStringList<ST> string_list)
{
	for (auto i : range(string_list.num_strings))
	{
		auto& vec = string_list.strings[i];
		auto decompressed = apply_to_string(vec.string, vec.slen);
		SG_FREE(vec.string);
		vec.string = decompressed;
	}
}

template <class ST>
ST* CDecompressString<ST>::apply_to_string(ST* f, int32_t &len)
{
	uint64_t compressed_size=((int32_t*) f)[0];
	uint64_t uncompressed_size=((int32_t*) f)[1];

	int32_t offs = std::ceil(2.0 * sizeof(int32_t) / sizeof(ST));
	ASSERT(uint64_t(len)==uint64_t(offs)+compressed_size)

	len=uncompressed_size;
	uncompressed_size*=sizeof(ST);
	ST* vec=SG_MALLOC(ST, len);
	compressor->decompress((uint8_t*) (&f[offs]), compressed_size,
			(uint8_t*) vec, uncompressed_size);

	ASSERT(uncompressed_size==((uint64_t) len)*sizeof(ST))
	return vec;
}

template <class ST>
EPreprocessorType CDecompressString<ST>::get_type() const
{
	return P_DECOMPRESSSTRING;
}

template class CDecompressString<bool>;
template class CDecompressString<char>;
template class CDecompressString<int8_t>;
template class CDecompressString<uint8_t>;
template class CDecompressString<int16_t>;
template class CDecompressString<uint16_t>;
template class CDecompressString<int32_t>;
template class CDecompressString<uint32_t>;
template class CDecompressString<int64_t>;
template class CDecompressString<uint64_t>;
template class CDecompressString<float32_t>;
template class CDecompressString<float64_t>;
template class CDecompressString<floatmax_t>;
}
