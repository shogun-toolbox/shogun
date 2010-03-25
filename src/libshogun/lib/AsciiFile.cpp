#include "lib/File.h"
#include "features/SparseFeatures.h"
#include "lib/AsciiFile.h"

using namespace shogun;

CAsciiFile::CAsciiFile(FILE* f, const char* name) : CFile(f, name)
{
}

CAsciiFile::CAsciiFile(char* fname, char rw, const char* name) : CFile(fname, rw, name)
{
}

CAsciiFile::~CAsciiFile()
{
}

void CAsciiFile::set_variable_name(const char* name)
{
}

char* CAsciiFile::get_variable_name()
{
	return NULL;
}

#define GET_VECTOR(fname, sg_type) \
void CAsciiFile::fname(sg_type*& vec, int32_t& len) \
{													\
	vec=NULL;										\
	len=0;											\
}

GET_VECTOR(get_byte_vector, uint8_t)
GET_VECTOR(get_char_vector, char)
GET_VECTOR(get_int_vector, int32_t)
GET_VECTOR(get_shortreal_vector, float32_t)
GET_VECTOR(get_real_vector, float64_t)
GET_VECTOR(get_short_vector, int16_t)
GET_VECTOR(get_word_vector, uint16_t)
#undef GET_VECTOR

#define GET_MATRIX(fname, conv, sg_type)										\
void CAsciiFile::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																				\
	struct stat stats;															\
	if (stat(filename, &stats)!=0)												\
		SG_ERROR("Could not get file statistics.\n");							\
																				\
	char* data=new char[stats.st_size+1];										\
	memset(data, 0, sizeof(char)*(stats.st_size+1));							\
	size_t nread=fread(data, sizeof(char), stats.st_size, file);				\
	if (nread<=0)																\
		SG_ERROR("Could not read data from %s.\n");								\
																				\
	SG_DEBUG("data read from file:\n%s\n", data);								\
																				\
	/* determine num_feat and num_vec, populate dynamic array */ 				\
	int32_t nf=0;																\
	num_feat=0;																	\
	num_vec=0;																	\
	char* ptr_item=NULL;														\
	char* ptr_data=data;														\
	CDynamicArray<char*>* items=new CDynamicArray<char*>();						\
																				\
	while (*ptr_data)															\
	{																			\
		if (*ptr_data=='\n')													\
		{																		\
			if (ptr_item)														\
				nf++;															\
																				\
			if (num_feat!=0 && nf!=num_feat)									\
				SG_ERROR("Number of features mismatches (%d != %d) in vector"	\
						" %d in file %s.\n", num_feat, nf, num_vec, filename);	\
																				\
			append_item(items, ptr_data, ptr_item);								\
			num_feat=nf;														\
			num_vec++;															\
			nf=0;																\
			ptr_item=NULL;														\
		}																		\
		else if (!isblank(*ptr_data) && !ptr_item)								\
		{																		\
			ptr_item=ptr_data;													\
		}																		\
		else if (isblank(*ptr_data) && ptr_item)								\
		{																		\
			append_item(items, ptr_data, ptr_item);								\
			ptr_item=NULL;														\
			nf++;																\
		}																		\
																				\
		ptr_data++;																\
	}																			\
																				\
	SG_DEBUG("num feat: %d, num_vec %d\n", num_feat, num_vec);					\
	delete[] data;																\
																				\
	/* now copy data into matrix */ 											\
	matrix=new sg_type[num_vec*num_feat];										\
	for (int32_t i=0; i<num_vec; i++)											\
	{																			\
		for (int32_t j=0; j<num_feat; j++)										\
		{																		\
			char* item=items->get_element(i*num_feat+j);						\
			matrix[i*num_feat+j]=atof(item);									\
			delete[] item;														\
		}																		\
	}																			\
	delete items;																\
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

void CAsciiFile::get_byte_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CAsciiFile::get_char_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CAsciiFile::get_int_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CAsciiFile::get_shortreal_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CAsciiFile::get_real_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CAsciiFile::get_short_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CAsciiFile::get_word_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CAsciiFile::get_real_sparsematrix(TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	/*const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read SPARSE REAL matrix.\n");

	CFile f((char*) filename, 'r', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to read SPARSE REAL matrix.\n", filename);

	if (!f.read_real_valued_sparse(matrix, num_feat, num_vec))
		SG_ERROR("Could not read SPARSE REAL data from %s.\n", filename);*/
}


void CAsciiFile::get_byte_string_list(T_STRING<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_char_string_list(T_STRING<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
/*
	const char* filename=get_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to read CHAR string list.\n");

	CFile f((char*) filename, 'r', F_CHAR);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to read CHAR string list.\n", filename);

	if (!f.read_char_valued_strings(strings, num_str, max_string_len))
		SG_ERROR("Could not read CHAR data from %s.\n", filename);

	for (int32_t i=0; i<num_str; i++)
		SG_PRINT("%s\n", strings[i].string);
*/
}

void CAsciiFile::get_int_string_list(T_STRING<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_uint_string_list(T_STRING<uint32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_short_string_list(T_STRING<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_word_string_list(T_STRING<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_long_string_list(T_STRING<int64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_ulong_string_list(T_STRING<uint64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_shortreal_string_list(T_STRING<float32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_real_string_list(T_STRING<float64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CAsciiFile::get_longreal_string_list(T_STRING<floatmax_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}


/** set functions - to pass data from shogun to the target interface */

void CAsciiFile::set_char_vector(const char* vec, int32_t len)
{
}

void CAsciiFile::set_short_vector(const int16_t* vec, int32_t len)
{
}

void CAsciiFile::set_byte_vector(const uint8_t* vec, int32_t len)
{
}

void CAsciiFile::set_int_vector(const int32_t* vec, int32_t len)
{
}

void CAsciiFile::set_shortreal_vector(const float32_t* vec, int32_t len)
{
}

void CAsciiFile::set_real_vector(const float64_t* vec, int32_t len)
{
/*	const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write REAL vector.\n");

	CFile f((char*) filename, 'w', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write REAL vector.\n", filename);

	if (!f.write_real_valued_dense(vec, len, 1))
		SG_ERROR("Could not write REAL data to %s.\n", filename);*/
}

void CAsciiFile::set_word_vector(const uint16_t* vec, int32_t len)
{
}

/*
#undef SET_VECTOR
#define SET_VECTOR(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CAsciiFile::function_name(const sg_type* vec, int32_t len)	\
{																\
	void* feat=NULL;												\
	PROTECT( feat = allocVector(r_type, len) );					\
																\
	for (int32_t i=0; i<len; i++)									\
		r_cast(feat)[i]=(if_type) vec[i];						\
																\
	UNPROTECT(1);												\
	set_arg_increment(feat);									\
}

SET_VECTOR(set_byte_vector, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_VECTOR(set_int_vector, INTSXP, INTEGER, int32_t, int, "Integer")
SET_VECTOR(set_short_vector, INTSXP, INTEGER, int16_t, int, "Short")
SET_VECTOR(set_shortreal_vector, XP, REAL, float32_t, float, "Single Precision")
SET_VECTOR(set_real_vector, XP, REAL, float64_t, double, "Double Precision")
SET_VECTOR(set_word_vector, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_VECTOR
*/


void CAsciiFile::set_char_matrix(const char* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_byte_matrix(const uint8_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_int_matrix(const int32_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_uint_matrix(const uint32_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_long_matrix(const int64_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_ulong_matrix(const uint64_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_short_matrix(const int16_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_shortreal_matrix(const float32_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_real_matrix(const float64_t* matrix, int32_t num_feat, int32_t num_vec)
{
	/*const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write REAL matrix.\n");

	CFile f((char*) filename, 'w', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write REAL matrix.\n", filename);

	if (!f.write_real_valued_dense(matrix, num_feat, num_vec))
		SG_ERROR("Could not write REAL data to %s.\n", filename);*/
}
void CAsciiFile::set_longreal_matrix(const floatmax_t* matrix, int32_t num_feat, int32_t num_vec)
{
}
void CAsciiFile::set_word_matrix(const uint16_t* matrix, int32_t num_feat, int32_t num_vec)
{
}

/*
#define SET_MATRIX(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CAsciiFile::function_name(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{																			\
	void* feat=NULL;															\
	PROTECT( feat = allocMatrix(r_type, num_feat, num_vec) );				\
																			\
	for (int32_t i=0; i<num_vec; i++)											\
	{																		\
		for (int32_t j=0; j<num_feat; j++)										\
			r_cast(feat)[i*num_feat+j]=(if_type) matrix[i*num_feat+j];		\
	}																		\
																			\
	UNPROTECT(1);															\
	set_arg_increment(feat);												\
}
SET_MATRIX(set_byte_matrix, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_MATRIX(set_int_matrix, INTSXP, INTEGER, int32_t, int, "Integer")
SET_MATRIX(set_short_matrix, INTSXP, INTEGER, int16_t, int, "Short")
SET_MATRIX(set_shortreal_matrix, XP, REAL, float32_t, float, "Single Precision")
SET_MATRIX(set_real_matrix, XP, REAL, float64_t, double, "Double Precision")
SET_MATRIX(set_word_matrix, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_MATRIX
*/


void CAsciiFile::set_real_sparsematrix(const TSparse<float64_t>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)
{
	/*const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write SPARSE REAL matrix.\n");

	CFile f((char*) filename, 'w', F_DREAL);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write SPARSE REAL matrix.\n", filename);

	if (!f.write_real_valued_sparse(matrix, num_feat, num_vec))
		SG_ERROR("Could not write SPARSE REAL data to %s.\n", filename);*/
}

void CAsciiFile::set_byte_string_list(const T_STRING<uint8_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_char_string_list(const T_STRING<char>* strings, int32_t num_str)
{
	/*const char* filename=set_arg_increment();
	if (!filename)
		SG_ERROR("No filename given to write CHAR string list.\n");

	CFile f((char*) filename, 'w', F_CHAR);
	if (!f.is_ok())
		SG_ERROR("Could not open file %s to write CHAR string list.\n", filename);

	if (!f.write_char_valued_strings(strings, num_str))
		SG_ERROR("Could not write CHAR data to %s.\n", filename);*/
}

void CAsciiFile::set_int_string_list(const T_STRING<int32_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_uint_string_list(const T_STRING<uint32_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_short_string_list(const T_STRING<int16_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_word_string_list(const T_STRING<uint16_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_long_string_list(const T_STRING<int64_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_ulong_string_list(const T_STRING<uint64_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_shortreal_string_list(const T_STRING<float32_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_real_string_list(const T_STRING<float64_t>* strings, int32_t num_str)
{
}

void CAsciiFile::set_longreal_string_list(const T_STRING<floatmax_t>* strings, int32_t num_str)
{
}

template <class T> void CAsciiFile::append_item(
	CDynamicArray<T>* items, char* ptr_data, char* ptr_item)
{
	size_t len=(ptr_data-ptr_item)/sizeof(char);
	char* item=new char[len+1];
	memset(item, 0, sizeof(char)*(len+1));
	item=strncpy(item, ptr_item, len);

	SG_DEBUG("current %c, len %d, item %s\n", *ptr_data, len, item);
	items->append_element(item);
}

bool CAsciiFile::read_real_valued_sparse(
	TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
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
	return true;
}

bool CAsciiFile::write_real_valued_sparse(
	const TSparse<float64_t>* matrix, int32_t num_feat, int32_t num_vec)
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

	return true;
}


bool CAsciiFile::read_char_valued_strings(
	T_STRING<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	bool result=false;

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
		result=true;
		SG_INFO("file successfully read\n");
		SG_INFO("max_string_length=%d\n", max_string_len);
		SG_INFO("num_strings=%d\n", num_str);
	}

	delete[] dummy;
	delete[] overflow;

	return result;
}

bool CAsciiFile::write_char_valued_strings(
	const T_STRING<char>* strings, int32_t num_str)
{
	if (!(file && strings))
		SG_ERROR("File or strings invalid.\n");

	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len = strings[i].length;
		fwrite(strings[i].string, sizeof(char), len, file);
		fprintf(file, "\n");
	}

	return true;
}



bool CAsciiFile::write_real_valued_dense(
	const float64_t* matrix, int32_t num_feat, int32_t num_vec)
{
	if (!(file && matrix))
		SG_ERROR("File or matrix invalid.\n");

	for (int32_t i=0; i<num_feat; i++)
	{
		for (int32_t j=0; j<num_vec; j++)
		{
			float64_t v=matrix[num_feat*j+i];
			if (j==num_vec-1)
				fprintf(file, "%f\n", v);
			else
				fprintf(file, "%f ", v);
		}
	}

	return true;
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
}

{
	bool status=false;

	delete[] labels;
	num_labels=0;

	CFile f(fname, 'r', F_DREAL);
	int64_t num_lab=0;
	labels=f.load_real_data(NULL, num_lab);
	num_labels=num_lab;

    if (!f.is_ok()) {
      SG_ERROR( "loading file \"%s\" failed", fname);
    }
	else
	{
		SG_INFO( "%ld labels successfully read\n", num_labels);
		status=true;
	}

	return status;
}
*/
