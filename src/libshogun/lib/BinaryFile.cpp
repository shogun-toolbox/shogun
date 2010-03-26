#include "lib/File.h"
#include "features/SparseFeatures.h"
#include "lib/BinaryFile.h"

using namespace shogun;

CBinaryFile::CBinaryFile(FILE* f, const char* name) : CFile(f, name)
{
}

CBinaryFile::CBinaryFile(char* fname, char rw, const char* name) : CFile(fname, rw, name)
{
}

CBinaryFile::~CBinaryFile()
{
}

void CBinaryFile::set_variable_name(const char* name)
{
}

char* CBinaryFile::get_variable_name()
{
	return NULL;
}

#define GET_VECTOR(fname, mfname, sg_type) \
void CBinaryFile::fname(sg_type*& vec, int32_t& len) \
{													\
	vec=NULL;										\
	len=0;											\
	int32_t num_feat=0;								\
	int32_t num_vec=0;								\
	mfname(vec, num_feat, num_vec);					\
	if ((num_feat==1) || (num_vec==1))				\
	{												\
		if (num_feat==1)							\
			len=num_vec;							\
		else										\
			len=num_feat;							\
	}												\
	else											\
	{												\
		delete[] vec;								\
		vec=NULL;									\
		len=0;										\
		SG_ERROR("Could not read vector from"		\
				" file %s (shape %dx%d found but "	\
				"vector expected).\n", filename,	\
				num_vec, num_feat);					\
	}												\
}

GET_VECTOR(get_byte_vector, get_byte_matrix, uint8_t)
GET_VECTOR(get_char_vector, get_char_matrix, char)
GET_VECTOR(get_int_vector, get_int_matrix, int32_t)
GET_VECTOR(get_shortreal_vector, get_shortreal_matrix, float32_t)
GET_VECTOR(get_real_vector, get_real_matrix, float64_t)
GET_VECTOR(get_short_vector, get_short_matrix, int16_t)
GET_VECTOR(get_word_vector, get_word_matrix, uint16_t)
#undef GET_VECTOR

#define GET_MATRIX(fname, conv, sg_type)										\
void CBinaryFile::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																				\
}

GET_MATRIX(get_byte_matrix, atoi, uint8_t)
GET_MATRIX(get_char_matrix, atoi, char)
GET_MATRIX(get_int_matrix, atoi, int32_t)
GET_MATRIX(get_uint_matrix, atoi, uint32_t)
GET_MATRIX(get_long_matrix, atoll, int64_t)
GET_MATRIX(get_ulong_matrix, atoll, uint64_t)
GET_MATRIX(get_shortreal_matrix, atof, float32_t)
GET_MATRIX(get_real_matrix, atof, float64_t)
GET_MATRIX(get_longreal_matrix, atof, floatmax_t)
GET_MATRIX(get_short_matrix, atoi, int16_t)
GET_MATRIX(get_word_matrix, atoi, uint16_t)
#undef GET_MATRIX

void CBinaryFile::get_byte_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CBinaryFile::get_char_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CBinaryFile::get_int_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CBinaryFile::get_shortreal_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CBinaryFile::get_real_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CBinaryFile::get_short_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CBinaryFile::get_word_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CBinaryFile::get_real_sparsematrix(TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	size_t blocksize=1024*1024;
	size_t required_blocksize=blocksize;
	uint8_t* dummy=new uint8_t[blocksize];

	if (file)
	{
		num_vec=0;
		num_feat=0;

		SG_INFO("counting line numbers in file %s\n", filename);
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(file, 0, SEEK_END);
		size_t fsize=ftell(file);
		rewind(file);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, file);
			bool contains_cr=false;
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_vec++;
					contains_cr=true;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs+1);
					old_block_offs=block_offs;
				}
			}
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t");
		}

		SG_INFO("found %d feature vectors\n", num_vec);
		delete[] dummy;
		blocksize=required_blocksize;
		dummy = new uint8_t[blocksize+1]; //allow setting of '\0' at EOL
		matrix=new TSparse<float64_t>[num_vec];

		rewind(file);
		sz=blocksize;
		int32_t lines=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, file);

			size_t old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (i==sz-1 && dummy[i]!='\n' && sz==blocksize)
				{
					size_t len=i-old_sz+1;
					uint8_t* data=&dummy[old_sz];

					for (size_t j=0; j<len; j++)
						dummy[j]=data[j];

					sz=fread(dummy+len, sizeof(uint8_t), blocksize-len, file);
					i=0;
					old_sz=0;
					sz+=len;
				}

				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{

					size_t len=i-old_sz;
					uint8_t* data=&dummy[old_sz];

					int32_t dims=0;
					for (size_t j=0; j<len; j++)
					{
						if (data[j]==':')
							dims++;
					}

					if (dims<=0)
					{
						SG_ERROR("Error in line %d - number of"
								" dimensions is %d line is %d characters"
								" long\n line_content:'%.*s'\n", lines,
								dims, len, len, (const char*) data);
					}

					TSparseEntry<float64_t>* feat=new TSparseEntry<float64_t>[dims];

					//skip label part
					size_t j=0;
					for (; j<len; j++)
					{
						if (data[j]==':')
						{
							j=-1; //file without label
							break;
						}

						if (data[j]==' ')
						{
							data[j]='\0';

							//skip label part
							break;
						}
					}

					int32_t d=0;
					j++;
					uint8_t* start=&data[j];
					for (; j<len; j++)
					{
						if (data[j]==':')
						{
							data[j]='\0';

							feat[d].feat_index=(int32_t) atoi((const char*) start)-1;
							num_feat=CMath::max(num_feat, feat[d].feat_index+1);

							j++;
							start=&data[j];
							for (; j<len; j++)
							{
								if (data[j]==' ' || data[j]=='\n')
								{
									data[j]='\0';
									feat[d].entry=(float64_t) atof((const char*) start);
									d++;
									break;
								}
							}

							if (j==len)
							{
								data[j]='\0';
								feat[dims-1].entry=(float64_t) atof((const char*) start);
							}

							j++;
							start=&data[j];
						}
					}

					matrix[lines].vec_index=lines;
					matrix[lines].num_feat_entries=dims;
					matrix[lines].features=feat;

					old_sz=i+1;
					lines++;
					SG_PROGRESS(lines, 0, num_vec, 1, "LOADING:\t");
				}
			}
		}

		SG_INFO("file successfully read\n");
	}

	delete[] dummy;
}


void CBinaryFile::get_byte_string_list(T_STRING<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_char_string_list(T_STRING<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	size_t blocksize=1024*1024;
	size_t required_blocksize=0;
	char* dummy=new char[blocksize];
	char* overflow=NULL;
	int32_t overflow_len=0;

	if (file)
	{
		num_str=0;
		max_string_len=0;

		SG_INFO("counting line numbers in file %s\n", filename);
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(file, 0, SEEK_END);
		size_t fsize=ftell(file);
		rewind(file);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(char), blocksize, file);
			bool contains_cr=false;
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_str++;
					contains_cr=true;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs);
					old_block_offs=block_offs;
				}
			}
			SG_PROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t");
		}

		SG_INFO("found %d strings\n", num_str);
		SG_DEBUG("block_size=%d\n", required_blocksize);
		delete[] dummy;
		blocksize=required_blocksize;
		dummy=new char[blocksize];
		overflow=new char[blocksize];
		strings=new T_STRING<char>[num_str];

		rewind(file);
		sz=blocksize;
		int32_t lines=0;
		size_t old_sz=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(char), blocksize, file);

			old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					int32_t len=i-old_sz;
					max_string_len=CMath::max(max_string_len, len+overflow_len);

					strings[lines].length=len+overflow_len;
					strings[lines].string=new char[len+overflow_len];

					for (int32_t j=0; j<overflow_len; j++)
						strings[lines].string[j]=overflow[j];
					for (int32_t j=0; j<len; j++)
						strings[lines].string[j+overflow_len]=dummy[old_sz+j];

					// clear overflow
					overflow_len=0;

					//CMath::display_vector(strings[lines].string, len);
					old_sz=i+1;
					lines++;
					SG_PROGRESS(lines, 0, num_str, 1, "LOADING:\t");
				}
			}

			for (size_t i=old_sz; i<sz; i++)
				overflow[i-old_sz]=dummy[i];

			overflow_len=sz-old_sz;
		}
		SG_INFO("file successfully read\n");
		SG_INFO("max_string_length=%d\n", max_string_len);
		SG_INFO("num_strings=%d\n", num_str);
	}

	delete[] dummy;
	delete[] overflow;
}

void CBinaryFile::get_int_string_list(T_STRING<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_uint_string_list(T_STRING<uint32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_short_string_list(T_STRING<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_word_string_list(T_STRING<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_long_string_list(T_STRING<int64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_ulong_string_list(T_STRING<uint64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_shortreal_string_list(T_STRING<float32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_real_string_list(T_STRING<float64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CBinaryFile::get_longreal_string_list(T_STRING<floatmax_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}


/** set functions - to pass data from shogun to the target interface */

#define SET_VECTOR(fname, mfname, sg_type)	\
void CBinaryFile::fname(const sg_type* vec, int32_t len)	\
{															\
	mfname(vec, len, 1);									\
}
SET_VECTOR(set_byte_vector, set_byte_matrix, uint8_t)
SET_VECTOR(set_char_vector, set_char_matrix, char)
SET_VECTOR(set_int_vector, set_int_matrix, int32_t)
SET_VECTOR(set_shortreal_vector, set_shortreal_matrix, float32_t)
SET_VECTOR(set_real_vector, set_real_matrix, float64_t)
SET_VECTOR(set_short_vector, set_short_matrix, int16_t)
SET_VECTOR(set_word_vector, set_word_matrix, uint16_t)
#undef SET_VECTOR

#define SET_MATRIX(fname, sg_type, fprt_type, type_str) \
void CBinaryFile::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																					\
	if (!(file && matrix))															\
		SG_ERROR("File or matrix invalid.\n");										\
																					\
	for (int32_t i=0; i<num_feat; i++)												\
	{																				\
		for (int32_t j=0; j<num_vec; j++)											\
		{																			\
			sg_type v=matrix[num_feat*j+i];											\
			if (j==num_vec-1)														\
				fprintf(file, type_str "\n", (fprt_type) v);						\
			else																	\
				fprintf(file, type_str " ", (fprt_type) v);							\
		}																			\
	}																				\
}
SET_MATRIX(set_char_matrix, char, char, "%c")
SET_MATRIX(set_byte_matrix, uint8_t, uint8_t, "%u")
SET_MATRIX(set_int_matrix, int32_t, int32_t, "%i")
SET_MATRIX(set_uint_matrix, uint32_t, uint32_t, "%u")
SET_MATRIX(set_long_matrix, int64_t, long long int, "%lli")
SET_MATRIX(set_ulong_matrix, uint64_t, long long unsigned int, "%llu")
SET_MATRIX(set_short_matrix, int16_t, int16_t, "%i")
SET_MATRIX(set_word_matrix, uint16_t, uint16_t, "%u")
SET_MATRIX(set_shortreal_matrix, float32_t, float32_t, "%f")
SET_MATRIX(set_real_matrix, float64_t, float64_t, "%f")
SET_MATRIX(set_longreal_matrix, floatmax_t, floatmax_t, "%Lf")
#undef SET_MATRIX

void CBinaryFile::set_real_sparsematrix(const TSparse<float64_t>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)
{
	if (!(file && matrix))
		SG_ERROR("File or matrix invalid.\n");

	for (int32_t i=0; i<num_vec; i++)
	{
		TSparseEntry<float64_t>* vec = matrix[i].features;
		int32_t len=matrix[i].num_feat_entries;

		for (int32_t j=0; j<len; j++)
		{
			if (j<len-1)
				fprintf(file, "%d:%f ", (int32_t) vec[j].feat_index+1, (double) vec[j].entry);
			else
				fprintf(file, "%d:%f\n", (int32_t) vec[j].feat_index+1, (double) vec[j].entry);
		}
	}
}

void CBinaryFile::set_byte_string_list(const T_STRING<uint8_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_char_string_list(const T_STRING<char>* strings, int32_t num_str)
{
	if (!(file && strings))
		SG_ERROR("File or strings invalid.\n");

	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len = strings[i].length;
		fwrite(strings[i].string, sizeof(char), len, file);
		fprintf(file, "\n");
	}
}

void CBinaryFile::set_int_string_list(const T_STRING<int32_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_uint_string_list(const T_STRING<uint32_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_short_string_list(const T_STRING<int16_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_word_string_list(const T_STRING<uint16_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_long_string_list(const T_STRING<int64_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_ulong_string_list(const T_STRING<uint64_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_shortreal_string_list(const T_STRING<float32_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_real_string_list(const T_STRING<float64_t>* strings, int32_t num_str)
{
}

void CBinaryFile::set_longreal_string_list(const T_STRING<floatmax_t>* strings, int32_t num_str)
{
}

/*
bool load(char* fname)
{
			int64_t length=0;
			max_string_length=0;

			CFile f(fname, 'r', F_CHAR);
			char* feature_matrix=f.load_char_data(NULL, length);

			SG_DEBUG("char data now at %p of length %ld\n", 
					feature_matrix, (int64_t) length);

			num_vectors=0;

			if (f.is_ok())
			{
				for (int64_t i=0; i<length; i++)
				{
					if (feature_matrix[i]=='\n')
						num_vectors++;
				}

				SG_INFO( "file contains %ld vectors\n", num_vectors);
				features= new T_STRING<ST>[num_vectors];

				int64_t index=0;
				for (int32_t lines=0; lines<num_vectors; lines++)
				{
					char* p=&feature_matrix[index];
					int32_t columns=0;

					for (columns=0; index+columns<length && p[columns]!='\n'; columns++);

					if (index+columns>=length && p[columns]!='\n') {
						SG_ERROR( "error in \"%s\":%d\n", fname, lines);
					}

					features[lines].length=columns;
					features[lines].string=new ST[columns];

					max_string_length=CMath::max(max_string_length,columns);

					for (int32_t i=0; i<columns; i++)
						features[lines].string[i]= ((ST) p[i]);

					index+= features[lines].length+1;
				}

				num_symbols=4; //FIXME
				return true;
			}
			else
				SG_ERROR( "reading file failed\n");

			return false;
} */

/*
int32_t* CFile::load_int_data(int32_t* target, int64_t& num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<int32_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_int_data(int32_t* src, int64_t num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<int32_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

float64_t* CFile::load_real_data(float64_t* target, int64_t& num)
{
	ASSERT(expected_type==F_DREAL);
	CSimpleFile<float64_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

float32_t* CFile::load_shortreal_data(float32_t* target, int64_t& num)
{
	ASSERT(expected_type==F_SHORTREAL);
	CSimpleFile<float32_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_real_data(float64_t* src, int64_t num)
{
	ASSERT(expected_type==F_DREAL);
	CSimpleFile<float64_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

bool CFile::save_shortreal_data(float32_t* src, int64_t num)
{
	ASSERT(expected_type==F_SHORTREAL);
	CSimpleFile<float32_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

char* CFile::load_char_data(char* target, int64_t& num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<char> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_char_data(char* src, int64_t num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<char> f(filename, file);
	status=f.save(src, num);
	return status;
}

uint8_t* CFile::load_byte_data(uint8_t* target, int64_t& num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<uint8_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_byte_data(uint8_t* src, int64_t num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<uint8_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

uint16_t* CFile::load_word_data(uint16_t* target, int64_t& num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<uint16_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_word_data(uint16_t* src, int64_t num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<uint16_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

int16_t* CFile::load_short_data(int16_t* target, int64_t& num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<int16_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_short_data(int16_t* src, int64_t num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<int16_t> f(filename, file);
	status=f.save(src, num);
	return status;
}
	int32_t i=0;
	int32_t num_left=get_num_vec_lhs();
	int32_t num_right=rhs->get_num_vectors();
	KERNELCACHE_IDX num_total=num_left*num_right;

	CFile f(fname, 'w', F_DREAL);

    for (int32_t l=0; l< (int32_t) num_left && f.is_ok(); l++)
	{
		for (int32_t r=0; r< (int32_t) num_right && f.is_ok(); r++)
		{
			 if (!(i % (num_total/200+1)))
				SG_PROGRESS(i, 0, num_total-1);

			float64_t k=kernel(l,r);
			f.save_real_data(&k, 1);

			i++;
		}
	}
	SG_DONE();

	if (f.is_ok())
		SG_INFO( "kernel matrix of size %ld x %ld written (filesize: %ld)\n", num_left, num_right, num_total*sizeof(KERNELCACHE_ELEM));

    return (f.is_ok());
*/
