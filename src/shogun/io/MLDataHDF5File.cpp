/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Zhengyang Liu (zhengyangl)
 */

#include <shogun/lib/config.h>

#ifdef HAVE_HDF5
#ifdef HAVE_CURL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <curl/curl.h>
#include <shogun/lib/memory.h>
#include <shogun/io/MLDataHDF5File.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/features/SparseFeatures.h>

using namespace shogun;

CMLDataHDF5File::CMLDataHDF5File()
{
	SG_UNSTABLE("CMLDataHDF5File::CMLDataHDF5File()", "\n")

	get_boolean_type();
	h5file = -1;
}

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
	size_t written = fwrite(ptr, size, nmemb, stream);
	return written;
}

CMLDataHDF5File::CMLDataHDF5File(char* data_name,
                                 const char* name,
                                 const char* url_prefix) : CFile()
{
	get_boolean_type();
	H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

	if (name)
		set_variable_name(name);

	CURL *curl;
	FILE *fp=NULL;

	char *mldata_url = (char*)malloc(strlen(url_prefix)+strlen(data_name)+1);
	*mldata_url=(char)NULL;
	strcat(mldata_url, url_prefix);
	strcat(mldata_url, data_name);

	fname = (char*)malloc(strlen((char*)"/tmp/")+strlen(data_name)+strlen((char*)".h5")+1);
	*fname=(char)NULL;
	strcat(fname, (char*) "/tmp/");
	strcat(fname, data_name);
	strcat(fname, (char*) ".h5");

	curl = curl_easy_init();
	fp = fopen(fname,"wb");

	if (!fp)
	{
		SG_ERROR("Could not open file '%s'\n", fname)
		return;
	}

	if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, mldata_url);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &write_data);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
		curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_perform(curl);
		curl_easy_cleanup(curl);
	}

	if(fp)
		fclose(fp);

	h5file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

	if (h5file<0)
		SG_ERROR("Could not open data repository '%s'\n", data_name)
}

CMLDataHDF5File::~CMLDataHDF5File()
{
	H5Fclose(h5file);
	remove(fname);
}

#define GET_VECTOR(fname, sg_type, datatype)										\
void CMLDataHDF5File::fname(sg_type*& vec, int32_t& len)							\
{																					\
	if (!h5file)																	\
		SG_ERROR("File invalid.\n")												\
																					\
	int32_t* dims;																	\
	int32_t ndims;																	\
	int64_t nelements;																\
	hid_t dataset=H5Dopen2(h5file, variable_name, H5P_DEFAULT);					\
	if (dataset<0)																	\
		SG_ERROR("Error opening data set\n")										\
	hid_t dtype=H5Dget_type(dataset);												\
	H5T_class_t t_class=H5Tget_class(dtype);										\
	TSGDataType t datatype; hid_t h5_type=get_compatible_type(t_class, &t);         \
	if (h5_type==-1)																\
	{																				\
		H5Dclose(dataset);															\
		SG_INFO("No compatible datatype found\n")									\
	}																				\
	get_dims(dataset, dims, ndims, nelements);										\
	if (!((ndims==2 && dims[0]==nelements && dims[1]==1) ||							\
			(ndims==2 && dims[0]==1 && dims[1]==nelements) ||						\
			(ndims==1 && dims[0]==nelements)))										\
		SG_ERROR("Error not a 1-dimensional vector (ndims=%d, dims[0]=%d)\n", ndims, dims[0])	\
	vec=SG_MALLOC(sg_type, nelements);														\
	len=nelements;																	\
	herr_t status = H5Dread(dataset, h5_type, H5S_ALL, 								\
			H5S_ALL, H5P_DEFAULT, vec);												\
	H5Dclose(dataset);																\
	H5Tclose(dtype);																\
	SG_FREE(dims);																	\
	if (status<0)																	\
	{																				\
		SG_FREE(vec);																\
		SG_ERROR("Error reading dataset\n")										\
	}																				\
}

GET_VECTOR(get_vector, bool, (CT_VECTOR, ST_NONE, PT_BOOL))
GET_VECTOR(get_vector, int8_t, (CT_VECTOR, ST_NONE, PT_INT8))
GET_VECTOR(get_vector, uint8_t, (CT_VECTOR, ST_NONE, PT_UINT8))
GET_VECTOR(get_vector, char, (CT_VECTOR, ST_NONE, PT_CHAR))
GET_VECTOR(get_vector, int32_t, (CT_VECTOR, ST_NONE, PT_INT32))
GET_VECTOR(get_vector, uint32_t, (CT_VECTOR, ST_NONE, PT_UINT32))
GET_VECTOR(get_vector, float32_t, (CT_VECTOR, ST_NONE, PT_FLOAT32))
GET_VECTOR(get_vector, float64_t, (CT_VECTOR, ST_NONE, PT_FLOAT64))
GET_VECTOR(get_vector, floatmax_t, (CT_VECTOR, ST_NONE, PT_FLOATMAX))
GET_VECTOR(get_vector, int16_t, (CT_VECTOR, ST_NONE, PT_INT16))
GET_VECTOR(get_vector, uint16_t, (CT_VECTOR, ST_NONE, PT_INT16))
GET_VECTOR(get_vector, int64_t, (CT_VECTOR, ST_NONE, PT_INT64))
GET_VECTOR(get_vector, uint64_t, (CT_VECTOR, ST_NONE, PT_UINT64))
#undef GET_VECTOR

#define GET_MATRIX(fname, sg_type, datatype)										\
void CMLDataHDF5File::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																					\
	if (!h5file)																	\
		SG_ERROR("File invalid.\n")												\
																					\
	int32_t* dims;																	\
	int32_t ndims;																	\
	int64_t nelements;																\
	hid_t dataset = H5Dopen2(h5file, variable_name, H5P_DEFAULT);					\
	if (dataset<0)																	\
		SG_ERROR("Error opening data set\n")										\
	hid_t dtype = H5Dget_type(dataset);												\
	H5T_class_t t_class=H5Tget_class(dtype);										\
	TSGDataType t datatype; hid_t h5_type=get_compatible_type(t_class, &t);	        \
	if (h5_type==-1)																\
	{																				\
		H5Dclose(dataset);															\
		SG_INFO("No compatible datatype found\n")									\
	}																				\
	get_dims(dataset, dims, ndims, nelements);										\
	if (ndims!=2)																	\
		SG_ERROR("Error not a 2-dimensional matrix\n")								\
	matrix=SG_MALLOC(sg_type, nelements);													\
	num_feat=dims[0];																\
	num_vec=dims[1];																\
	herr_t status = H5Dread(dataset, h5_type, H5S_ALL, 								\
			H5S_ALL, H5P_DEFAULT, matrix);											\
	H5Dclose(dataset);																\
	H5Tclose(dtype);																\
	SG_FREE(dims);																	\
	if (status<0)																	\
	{																				\
		SG_FREE(matrix);															\
		SG_ERROR("Error reading dataset\n")										\
	}																				\
}

GET_MATRIX(get_matrix, bool, (CT_MATRIX, ST_NONE, PT_BOOL))
GET_MATRIX(get_matrix, char, (CT_MATRIX, ST_NONE, PT_CHAR))
GET_MATRIX(get_matrix, uint8_t, (CT_MATRIX, ST_NONE, PT_UINT8))
GET_MATRIX(get_matrix, int32_t, (CT_MATRIX, ST_NONE, PT_INT32))
GET_MATRIX(get_matrix, uint32_t, (CT_MATRIX, ST_NONE, PT_INT32))
GET_MATRIX(get_matrix, int64_t, (CT_MATRIX, ST_NONE, PT_INT64))
GET_MATRIX(get_matrix, uint64_t, (CT_MATRIX, ST_NONE, PT_INT64))
GET_MATRIX(get_matrix, int16_t, (CT_MATRIX, ST_NONE, PT_INT16))
GET_MATRIX(get_matrix, uint16_t, (CT_MATRIX, ST_NONE, PT_INT16))
GET_MATRIX(get_matrix, float32_t, (CT_MATRIX, ST_NONE, PT_FLOAT32))
GET_MATRIX(get_matrix, float64_t, (CT_MATRIX, ST_NONE, PT_FLOAT64))
GET_MATRIX(get_matrix, floatmax_t, (CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef GET_MATRIX

void CMLDataHDF5File::get_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CMLDataHDF5File::get_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CMLDataHDF5File::get_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CMLDataHDF5File::get_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CMLDataHDF5File::get_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CMLDataHDF5File::get_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CMLDataHDF5File::get_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

#define GET_SPARSEMATRIX(fname, sg_type, datatype)										\
void CMLDataHDF5File::fname(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																						\
	if (!(file))																		\
		SG_ERROR("File invalid.\n")													\
}
GET_SPARSEMATRIX(get_sparse_matrix, bool, DT_SPARSE_BOOL)
GET_SPARSEMATRIX(get_sparse_matrix, char, DT_SPARSE_CHAR)
GET_SPARSEMATRIX(get_sparse_matrix, int8_t, DT_SPARSE_INT8)
GET_SPARSEMATRIX(get_sparse_matrix, uint8_t, DT_SPARSE_BYTE)
GET_SPARSEMATRIX(get_sparse_matrix, int32_t, DT_SPARSE_INT)
GET_SPARSEMATRIX(get_sparse_matrix, uint32_t, DT_SPARSE_UINT)
GET_SPARSEMATRIX(get_sparse_matrix, int64_t, DT_SPARSE_LONG)
GET_SPARSEMATRIX(get_sparse_matrix, uint64_t, DT_SPARSE_ULONG)
GET_SPARSEMATRIX(get_sparse_matrix, int16_t, DT_SPARSE_SHORT)
GET_SPARSEMATRIX(get_sparse_matrix, uint16_t, DT_SPARSE_WORD)
GET_SPARSEMATRIX(get_sparse_matrix, float32_t, DT_SPARSE_SHORTREAL)
GET_SPARSEMATRIX(get_sparse_matrix, float64_t, DT_SPARSE_REAL)
GET_SPARSEMATRIX(get_sparse_matrix, floatmax_t, DT_SPARSE_LONGREAL)
#undef GET_SPARSEMATRIX


#define GET_STRING_LIST(fname, sg_type, datatype)												\
void CMLDataHDF5File::fname(SGString<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len) \
{																								\
}

GET_STRING_LIST(get_string_list, bool, DT_STRING_BOOL)
GET_STRING_LIST(get_string_list, char, DT_STRING_CHAR)
GET_STRING_LIST(get_string_list, int8_t, DT_STRING_INT8)
GET_STRING_LIST(get_string_list, uint8_t, DT_STRING_BYTE)
GET_STRING_LIST(get_string_list, int32_t, DT_STRING_INT)
GET_STRING_LIST(get_string_list, uint32_t, DT_STRING_UINT)
GET_STRING_LIST(get_string_list, int64_t, DT_STRING_LONG)
GET_STRING_LIST(get_string_list, uint64_t, DT_STRING_ULONG)
GET_STRING_LIST(get_string_list, int16_t, DT_STRING_SHORT)
GET_STRING_LIST(get_string_list, uint16_t, DT_STRING_WORD)
GET_STRING_LIST(get_string_list, float32_t, DT_STRING_SHORTREAL)
GET_STRING_LIST(get_string_list, float64_t, DT_STRING_REAL)
GET_STRING_LIST(get_string_list, floatmax_t, DT_STRING_LONGREAL)
#undef GET_STRING_LIST

void CMLDataHDF5File::get_boolean_type()
{
	boolean_type=H5T_NATIVE_UCHAR;
	switch (sizeof(bool))
	{
		case 1:
			boolean_type = H5T_NATIVE_UCHAR;
			break;
		case 2:
			boolean_type = H5T_NATIVE_UINT16;
			break;
		case 4:
			boolean_type = H5T_NATIVE_UINT32;
			break;
		case 8:
			boolean_type = H5T_NATIVE_UINT64;
			break;
		default:
			SG_ERROR("Boolean type not supported on this platform\n")
	}
}

hid_t CMLDataHDF5File::get_compatible_type(H5T_class_t t_class,
									 const TSGDataType* datatype)
{
	switch (t_class)
	{
		case H5T_FLOAT:
		case H5T_INTEGER:
			switch (datatype->m_ptype)
			{
			case PT_BOOL: return boolean_type;
			case PT_CHAR: return H5T_NATIVE_CHAR;
			case PT_INT8: return H5T_NATIVE_INT8;
			case PT_UINT8: return H5T_NATIVE_UINT8;
			case PT_INT16: return H5T_NATIVE_INT16;
			case PT_UINT16: return H5T_NATIVE_UINT16;
			case PT_INT32: return H5T_NATIVE_INT32;
			case PT_UINT32: return H5T_NATIVE_UINT32;
			case PT_INT64: return H5T_NATIVE_INT64;
			case PT_UINT64: return H5T_NATIVE_UINT64;
			case PT_FLOAT32: return H5T_NATIVE_FLOAT;
			case PT_FLOAT64: return H5T_NATIVE_DOUBLE;
			case PT_FLOATMAX: return H5T_NATIVE_LDOUBLE;
			case PT_COMPLEX64:
				SG_ERROR("complex64_t not compatible with HDF5File!");
				return -1;
			case PT_SGOBJECT:
				SG_ERROR("Implementation error during writing "
						 "HDF5File!");
				return -1;
			}
		case H5T_STRING:
			SG_ERROR("Strings not supported")
			return -1;
		case H5T_VLEN:
			SG_ERROR("Variable length containers currently not supported")
			return -1;
		case H5T_ARRAY:
			SG_ERROR("Array containers currently not supported")
			return -1;
		default:
			SG_ERROR("Datatype mismatchn")
			return -1;
	}
}

void CMLDataHDF5File::get_dims(hid_t dataset, int32_t*& dims, int32_t& ndims, int64_t& total_elements)
{
	hid_t dataspace = H5Dget_space(dataset);
	if (dataspace<0)
		SG_ERROR("Error obtaining hdf5 dataspace\n")

	ndims = H5Sget_simple_extent_ndims(dataspace);
	total_elements=H5Sget_simple_extent_npoints(dataspace);
	hsize_t* dims_out=SG_MALLOC(hsize_t, ndims);
	dims=SG_MALLOC(int32_t, ndims);
	H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
	for (int32_t i=0; i<ndims; i++)
		dims[i]=dims_out[i];
	SG_FREE(dims_out);
	H5Sclose(dataspace);
}

void CMLDataHDF5File::create_group_hierarchy()
{
	char* vname=get_strdup(variable_name);
	int32_t vlen=strlen(vname);
	for (int32_t i=0; i<vlen; i++)
	{
		if (i!=0 && vname[i]=='/')
		{
			vname[i]='\0';
			hid_t g = H5Gopen2(h5file, vname, H5P_DEFAULT);
			if (g<0)
			{
				g=H5Gcreate2(h5file, vname, H5P_DEFAULT, H5P_DEFAULT,
						H5P_DEFAULT);
				if (g<0)
					SG_ERROR("Error creating group '%s'\n", vname)
				vname[i]='/';
			}
			H5Gclose(g);
		}
	}
	SG_FREE(vname);
}
#endif //  CURL
#endif //  HDF5
