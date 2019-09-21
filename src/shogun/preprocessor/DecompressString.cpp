#include <shogun/preprocessor/DecompressString.h>

namespace shogun
{

/** default constructor  */
template <class ST>
DecompressString<ST>::DecompressString() : StringPreprocessor<ST>()
{
	SGObject::set_generic<ST>();
}

template <class ST>
DecompressString<ST>::DecompressString(E_COMPRESSION_TYPE ct) : StringPreprocessor<ST>()
{
	compressor=std::make_shared<Compressor>(ct);
	SGObject::set_generic<ST>();
}

template <class ST>
DecompressString<ST>::~DecompressString()
{
}

template <class ST>
void DecompressString<ST>::cleanup()
{
}

template <class ST>
bool DecompressString<ST>::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

template <class ST>
bool DecompressString<ST>::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

template <class ST>
void DecompressString<ST>::apply_to_string_list(std::vector<SGVector<ST>>& string_list)
{
	for (auto& vec : string_list)
		vec = SGVector<ST>(apply_to_string(vec.vector, vec.vlen), vec.vlen);
}

template <class ST>
ST* DecompressString<ST>::apply_to_string(ST* f, int32_t &len)
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
EPreprocessorType DecompressString<ST>::get_type() const
{
	return P_DECOMPRESSSTRING;
}

template class DecompressString<bool>;
template class DecompressString<char>;
template class DecompressString<int8_t>;
template class DecompressString<uint8_t>;
template class DecompressString<int16_t>;
template class DecompressString<uint16_t>;
template class DecompressString<int32_t>;
template class DecompressString<uint32_t>;
template class DecompressString<int64_t>;
template class DecompressString<uint64_t>;
template class DecompressString<float32_t>;
template class DecompressString<float64_t>;
template class DecompressString<floatmax_t>;
}
