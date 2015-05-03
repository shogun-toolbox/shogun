/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */
#include <shogun/io/ProtobufFile.h>

#ifdef HAVE_PROTOBUF

#include <shogun/io/SGIO.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGString.h>
#include <google/protobuf/message.h>

using namespace shogun;

CProtobufFile::CProtobufFile()
{
	init();
}

CProtobufFile::CProtobufFile(FILE* f, const char* name) :
	CFile(f, name)
{
	init();
}

CProtobufFile::CProtobufFile(const char* fname, char rw, const char* name) :
	CFile(fname, rw, name)
{
	init();
}

CProtobufFile::~CProtobufFile()
{
	SG_FREE(buffer);
}

void CProtobufFile::init()
{
	version=1;
	message_size=1024*1024;

	buffer=SG_MALLOC(uint8_t, message_size*sizeof(uint32_t));
}

#define GET_VECTOR(sg_type) \
void CProtobufFile::get_vector(sg_type*& vector, int32_t& len) \
{ \
	read_and_validate_global_header(ShogunVersion::VECTOR); \
	VectorHeader data_header=read_vector_header(); \
	len=data_header.len(); \
	read_memory_block(vector, len, data_header.num_messages()); \
}

GET_VECTOR(int8_t)
GET_VECTOR(uint8_t)
GET_VECTOR(char)
GET_VECTOR(int32_t)
GET_VECTOR(uint32_t)
GET_VECTOR(float32_t)
GET_VECTOR(float64_t)
GET_VECTOR(floatmax_t)
GET_VECTOR(int16_t)
GET_VECTOR(uint16_t)
GET_VECTOR(int64_t)
GET_VECTOR(uint64_t)
#undef GET_VECTOR

#define GET_MATRIX(read_func, sg_type) \
void CProtobufFile::get_matrix(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec) \
{ \
	read_and_validate_global_header(ShogunVersion::MATRIX); \
	MatrixHeader data_header=read_matrix_header(); \
	num_feat=data_header.num_cols(); \
	num_vec=data_header.num_rows(); \
	read_memory_block(matrix, num_feat*num_vec, data_header.num_messages()); \
}

GET_MATRIX(read_char, int8_t)
GET_MATRIX(read_byte, uint8_t)
GET_MATRIX(read_char, char)
GET_MATRIX(read_int, int32_t)
GET_MATRIX(read_uint, uint32_t)
GET_MATRIX(read_short_real, float32_t)
GET_MATRIX(read_real, float64_t)
GET_MATRIX(read_long_real, floatmax_t)
GET_MATRIX(read_short, int16_t)
GET_MATRIX(read_word, uint16_t)
GET_MATRIX(read_long, int64_t)
GET_MATRIX(read_ulong, uint64_t)
#undef GET_MATRIX

#define GET_NDARRAY(read_func, sg_type) \
void CProtobufFile::get_ndarray(sg_type*& array, int32_t*& dims, int32_t& num_dims) \
{ \
	SG_NOTIMPLEMENTED \
}

GET_NDARRAY(read_byte, uint8_t)
GET_NDARRAY(read_char, char)
GET_NDARRAY(read_int, int32_t)
GET_NDARRAY(read_short_real, float32_t)
GET_NDARRAY(read_real, float64_t)
GET_NDARRAY(read_short, int16_t)
GET_NDARRAY(read_word, uint16_t)
#undef GET_NDARRAY

#define GET_SPARSE_MATRIX(sg_type) \
void CProtobufFile::get_sparse_matrix( \
			SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec) \
{ \
	read_and_validate_global_header(ShogunVersion::SPARSE_MATRIX); \
	SparseMatrixHeader data_header=read_sparse_matrix_header(); \
	num_feat=data_header.num_features(); \
	num_vec=data_header.num_vectors(); \
	read_sparse_matrix(matrix, data_header); \
}

GET_SPARSE_MATRIX(bool)
GET_SPARSE_MATRIX(int8_t)
GET_SPARSE_MATRIX(uint8_t)
GET_SPARSE_MATRIX(char)
GET_SPARSE_MATRIX(int32_t)
GET_SPARSE_MATRIX(uint32_t)
GET_SPARSE_MATRIX(float32_t)
GET_SPARSE_MATRIX(float64_t)
GET_SPARSE_MATRIX(floatmax_t)
GET_SPARSE_MATRIX(int16_t)
GET_SPARSE_MATRIX(uint16_t)
GET_SPARSE_MATRIX(int64_t)
GET_SPARSE_MATRIX(uint64_t)
#undef GET_SPARSE_MATRIX

#define SET_VECTOR(sg_type) \
void CProtobufFile::set_vector(const sg_type* vector, int32_t len) \
{ \
	int32_t num_messages=compute_num_messages(len, sizeof(sg_type)); \
	write_global_header(ShogunVersion::VECTOR); \
	write_vector_header(len, num_messages); \
	write_memory_block(vector, len, num_messages); \
}

SET_VECTOR(int8_t)
SET_VECTOR(uint8_t)
SET_VECTOR(char)
SET_VECTOR(int32_t)
SET_VECTOR(uint32_t)
SET_VECTOR(int64_t)
SET_VECTOR(uint64_t)
SET_VECTOR(float32_t)
SET_VECTOR(float64_t)
SET_VECTOR(floatmax_t)
SET_VECTOR(int16_t)
SET_VECTOR(uint16_t)
#undef SET_VECTOR

#define SET_MATRIX(sg_type) \
void CProtobufFile::set_matrix(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	int32_t num_messages=compute_num_messages(num_feat*num_vec, sizeof(sg_type)); \
	write_global_header(ShogunVersion::MATRIX); \
	write_matrix_header(num_feat, num_vec, num_messages); \
	write_memory_block(matrix, num_feat*num_vec, num_messages); \
}

SET_MATRIX(int8_t)
SET_MATRIX(uint8_t)
SET_MATRIX(char)
SET_MATRIX(int32_t)
SET_MATRIX(uint32_t)
SET_MATRIX(int64_t)
SET_MATRIX(uint64_t)
SET_MATRIX(float32_t)
SET_MATRIX(float64_t)
SET_MATRIX(floatmax_t)
SET_MATRIX(int16_t)
SET_MATRIX(uint16_t)
#undef SET_MATRIX

#define SET_SPARSE_MATRIX(sg_type) \
void CProtobufFile::set_sparse_matrix( \
			const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	write_global_header(ShogunVersion::SPARSE_MATRIX); \
	write_sparse_matrix_header(matrix, num_feat, num_vec); \
	write_sparse_matrix(matrix, num_vec); \
}

SET_SPARSE_MATRIX(bool)
SET_SPARSE_MATRIX(int8_t)
SET_SPARSE_MATRIX(uint8_t)
SET_SPARSE_MATRIX(char)
SET_SPARSE_MATRIX(int32_t)
SET_SPARSE_MATRIX(uint32_t)
SET_SPARSE_MATRIX(int64_t)
SET_SPARSE_MATRIX(uint64_t)
SET_SPARSE_MATRIX(float32_t)
SET_SPARSE_MATRIX(float64_t)
SET_SPARSE_MATRIX(floatmax_t)
SET_SPARSE_MATRIX(int16_t)
SET_SPARSE_MATRIX(uint16_t)
#undef SET_SPARSE_MATRIX

#define GET_STRING_LIST(sg_type) \
void CProtobufFile::get_string_list( \
			SGString<sg_type>*& strings, int32_t& num_str, \
			int32_t& max_string_len) \
{ \
	read_and_validate_global_header(ShogunVersion::STRING_LIST); \
	StringListHeader data_header=read_string_list_header(); \
	num_str=data_header.num_str(); \
	max_string_len=data_header.max_string_len(); \
	read_string_list(strings, data_header); \
}

GET_STRING_LIST(int8_t)
GET_STRING_LIST(uint8_t)
GET_STRING_LIST(char)
GET_STRING_LIST(int32_t)
GET_STRING_LIST(uint32_t)
GET_STRING_LIST(int64_t)
GET_STRING_LIST(uint64_t)
GET_STRING_LIST(float32_t)
GET_STRING_LIST(float64_t)
GET_STRING_LIST(floatmax_t)
GET_STRING_LIST(int16_t)
GET_STRING_LIST(uint16_t)
#undef GET_STRING_LIST

#define SET_STRING_LIST(sg_type) \
void CProtobufFile::set_string_list( \
			const SGString<sg_type>* strings, int32_t num_str) \
{ \
	write_global_header(ShogunVersion::STRING_LIST); \
	write_string_list_header(strings, num_str); \
	write_string_list(strings, num_str); \
}

SET_STRING_LIST(int8_t)
SET_STRING_LIST(uint8_t)
SET_STRING_LIST(char)
SET_STRING_LIST(int32_t)
SET_STRING_LIST(uint32_t)
SET_STRING_LIST(int64_t)
SET_STRING_LIST(uint64_t)
SET_STRING_LIST(float32_t)
SET_STRING_LIST(float64_t)
SET_STRING_LIST(floatmax_t)
SET_STRING_LIST(int16_t)
SET_STRING_LIST(uint16_t)
#undef SET_STRING_LIST

void CProtobufFile::write_big_endian_uint(uint32_t number, uint8_t* array, uint32_t size)
{
	if (size<4)
		SG_ERROR("array is too small to write\n");

	array[0]=(number>>24)&0xffu;
	array[1]=(number>>16)&0xffu;
	array[2]=(number>>8)&0xffu;
	array[3]=number&0xffu;
}

uint32_t CProtobufFile::read_big_endian_uint(uint8_t* array, uint32_t size)
{
	if (size<4)
		SG_ERROR("array is too small to read\n");

	return (array[0]<<24) | (array[1]<<16) | (array[2]<<8) | array[3];
}

int32_t CProtobufFile::compute_num_messages(uint64_t len, int32_t sizeof_type) const
{
	uint32_t elements_in_message=message_size/sizeof_type;
	uint32_t num_messages=len/elements_in_message;
	if (len % elements_in_message > 0)
		num_messages++;

	return num_messages;
}

void CProtobufFile::read_and_validate_global_header(ShogunVersion_SGDataType type)
{
	ShogunVersion header;
	read_message(header);
	REQUIRE(header.version()==version, "wrong version\n")
	REQUIRE(header.data_type()==type, "wrong type\n")
}

void CProtobufFile::write_global_header(ShogunVersion_SGDataType type)
{
	ShogunVersion header;
	header.set_version(version);
	header.set_data_type(type);
	write_message(header);
}

VectorHeader CProtobufFile::read_vector_header()
{
	VectorHeader data_header;
	read_message(data_header);

	return data_header;
}

SparseMatrixHeader CProtobufFile::read_sparse_matrix_header()
{
	SparseMatrixHeader data_header;
	read_message(data_header);

	return data_header;
}

MatrixHeader CProtobufFile::read_matrix_header()
{
	MatrixHeader data_header;
	read_message(data_header);

	return data_header;
}

StringListHeader CProtobufFile::read_string_list_header()
{
	StringListHeader data_header;
	read_message(data_header);

	return data_header;
}

void CProtobufFile::write_vector_header(int32_t len, int32_t num_messages)
{
	VectorHeader data_header;
	data_header.set_len(len);
	data_header.set_num_messages(num_messages);
	write_message(data_header);
}

void CProtobufFile::write_matrix_header(int32_t num_feat, int32_t num_vec, int32_t num_messages)
{
	MatrixHeader data_header;
	data_header.set_num_cols(num_feat);
	data_header.set_num_rows(num_vec);
	data_header.set_num_messages(num_messages);
	write_message(data_header);
}

#define WRITE_SPARSE_MATRIX_HEADER(sg_type) \
void CProtobufFile::write_sparse_matrix_header( \
	const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec) \
{ \
	SparseMatrixHeader data_header; \
	data_header.set_num_features(num_feat); \
	data_header.set_num_vectors(num_vec); \
	for (int32_t i=0; i<num_vec; i++) \
	{ \
		data_header.add_num_feat_entries(matrix[i].num_feat_entries); \
	} \
	\
	write_message(data_header); \
}

WRITE_SPARSE_MATRIX_HEADER(bool)
WRITE_SPARSE_MATRIX_HEADER(int8_t)
WRITE_SPARSE_MATRIX_HEADER(uint8_t)
WRITE_SPARSE_MATRIX_HEADER(char)
WRITE_SPARSE_MATRIX_HEADER(int32_t)
WRITE_SPARSE_MATRIX_HEADER(uint32_t)
WRITE_SPARSE_MATRIX_HEADER(int64_t)
WRITE_SPARSE_MATRIX_HEADER(uint64_t)
WRITE_SPARSE_MATRIX_HEADER(float32_t)
WRITE_SPARSE_MATRIX_HEADER(float64_t)
WRITE_SPARSE_MATRIX_HEADER(floatmax_t)
WRITE_SPARSE_MATRIX_HEADER(int16_t)
WRITE_SPARSE_MATRIX_HEADER(uint16_t)
#undef WRITE_SPARSE_MATRIX_HEADER

#define WRITE_STRING_LIST_HEADER(sg_type) \
void CProtobufFile::write_string_list_header(const SGString<sg_type>* strings, int32_t num_str) \
{ \
	int32_t max_string_len=0; \
	StringListHeader data_header; \
	data_header.set_num_str(num_str); \
	for (int32_t i=0; i<num_str; i++) \
	{ \
		data_header.add_str_len(strings[i].slen); \
		if (strings[i].slen>max_string_len) \
			max_string_len=strings[i].slen; \
	} \
	data_header.set_max_string_len(max_string_len); \
	write_message(data_header); \
}

WRITE_STRING_LIST_HEADER(int8_t)
WRITE_STRING_LIST_HEADER(uint8_t)
WRITE_STRING_LIST_HEADER(char)
WRITE_STRING_LIST_HEADER(int32_t)
WRITE_STRING_LIST_HEADER(uint32_t)
WRITE_STRING_LIST_HEADER(int64_t)
WRITE_STRING_LIST_HEADER(uint64_t)
WRITE_STRING_LIST_HEADER(float32_t)
WRITE_STRING_LIST_HEADER(float64_t)
WRITE_STRING_LIST_HEADER(floatmax_t)
WRITE_STRING_LIST_HEADER(int16_t)
WRITE_STRING_LIST_HEADER(uint16_t)
#undef WRITE_STRING_LIST_HEADER

void CProtobufFile::read_message(google::protobuf::Message& message)
{
	uint32_t bytes_read=0;
	uint32_t msg_size=0;

	// read size of message
	bytes_read=fread(uint_buffer, sizeof(char), sizeof(uint32_t), file);
	REQUIRE(bytes_read==sizeof(uint32_t), "IO error\n");
	msg_size=read_big_endian_uint(uint_buffer, sizeof(uint32_t));
	REQUIRE(msg_size>0, "message size should be more than zero\n");

	// read message
	bytes_read=fread(buffer, sizeof(char), msg_size, file);
	REQUIRE(bytes_read==msg_size, "IO error\n");

	// try to parse message from read data
	REQUIRE(message.ParseFromArray(buffer, msg_size), "cannot parse header\n");
}

void CProtobufFile::write_message(const google::protobuf::Message& message)
{
	uint32_t bytes_write=0;
	uint32_t msg_size=message.ByteSize();

	// write size of message
	write_big_endian_uint(msg_size, uint_buffer, sizeof(uint32_t));
	bytes_write=fwrite(uint_buffer, sizeof(char), sizeof(uint32_t), file);
	REQUIRE(bytes_write==sizeof(uint32_t), "IO error\n");

	// write serialized message
	message.SerializeToArray(buffer, msg_size);
	bytes_write=fwrite(buffer, sizeof(char), msg_size, file);
	REQUIRE(bytes_write==msg_size, "IO error\n");
}

#define READ_MEMORY_BLOCK(chunk_type, sg_type) \
void CProtobufFile::read_memory_block(sg_type*& vector, uint64_t len, int32_t num_messages) \
{ \
	vector=SG_MALLOC(sg_type, len); \
	\
	chunk_type chunk; \
	int32_t elements_in_message=message_size/sizeof(sg_type); \
	for (int32_t i=0; i<num_messages; i++) \
	{ \
		read_message(chunk); \
		\
		int32_t num_elements_to_read=0; \
		if ((len-(i+1)*elements_in_message)<=0) \
			num_elements_to_read=len-i*elements_in_message; \
		else \
			num_elements_to_read=elements_in_message; \
		\
		for (int32_t j=0; j<num_elements_to_read; j++) \
			vector[j+i*elements_in_message]=chunk.data(j); \
	} \
}

READ_MEMORY_BLOCK(Int32Chunk, int8_t)
READ_MEMORY_BLOCK(UInt32Chunk, uint8_t)
READ_MEMORY_BLOCK(UInt32Chunk, char)
READ_MEMORY_BLOCK(Int32Chunk, int32_t)
READ_MEMORY_BLOCK(UInt32Chunk, uint32_t)
READ_MEMORY_BLOCK(Float32Chunk, float32_t)
READ_MEMORY_BLOCK(Float64Chunk, float64_t)
READ_MEMORY_BLOCK(Float64Chunk, floatmax_t)
READ_MEMORY_BLOCK(Int32Chunk, int16_t)
READ_MEMORY_BLOCK(UInt32Chunk, uint16_t)
READ_MEMORY_BLOCK(Int64Chunk, int64_t)
READ_MEMORY_BLOCK(UInt64Chunk, uint64_t)
#undef READ_MEMORY_BLOCK

#define WRITE_MEMORY_BLOCK(chunk_type, sg_type) \
void CProtobufFile::write_memory_block(const sg_type* vector, uint64_t len, int32_t num_messages) \
{ \
	chunk_type chunk; \
	int32_t elements_in_message=message_size/sizeof(sg_type); \
	for (int32_t i=0; i<num_messages; i++) \
	{ \
		\
		int32_t num_elements_to_write=0; \
		if ((len-(i+1)*elements_in_message)<=0) \
			num_elements_to_write=len-i*elements_in_message; \
		else \
			num_elements_to_write=elements_in_message; \
		\
		for (int32_t j=0; j<num_elements_to_write; j++) \
			chunk.add_data(vector[j+i*elements_in_message]); \
		\
		write_message(chunk); \
		chunk.Clear(); \
	} \
}

WRITE_MEMORY_BLOCK(Int32Chunk, int8_t)
WRITE_MEMORY_BLOCK(UInt32Chunk, uint8_t)
WRITE_MEMORY_BLOCK(UInt32Chunk, char)
WRITE_MEMORY_BLOCK(Int32Chunk, int32_t)
WRITE_MEMORY_BLOCK(UInt64Chunk, uint32_t)
WRITE_MEMORY_BLOCK(Int64Chunk, int64_t)
WRITE_MEMORY_BLOCK(UInt64Chunk, uint64_t)
WRITE_MEMORY_BLOCK(Float32Chunk, float32_t)
WRITE_MEMORY_BLOCK(Float64Chunk, float64_t)
WRITE_MEMORY_BLOCK(Float64Chunk, floatmax_t)
WRITE_MEMORY_BLOCK(Int32Chunk, int16_t)
WRITE_MEMORY_BLOCK(UInt32Chunk, uint16_t)
#undef WRITE_MEMORY_BLOCK

#define READ_SPARSE_MATRIX(chunk_type, sg_type) \
void CProtobufFile::read_sparse_matrix( \
			SGSparseVector<sg_type>*& matrix, const SparseMatrixHeader& data_header) \
{ \
	matrix=SG_MALLOC(SGSparseVector<sg_type>, data_header.num_vectors()); \
	\
	UInt64Chunk feat_index_chunk; \
	chunk_type entry_chunk; \
	read_message(feat_index_chunk); \
	read_message(entry_chunk); \
	\
	int32_t elements_in_message=message_size/sizeof(sg_type); \
	int32_t buffer_counter=0; \
	for (uint32_t i=0; i<data_header.num_vectors(); i++) \
	{ \
		matrix[i]=SGSparseVector<sg_type>(data_header.num_feat_entries(i)); \
		for (int32_t j=0; j<matrix[i].num_feat_entries; j++) \
		{ \
			matrix[i].features[j].feat_index=feat_index_chunk.data(buffer_counter); \
			matrix[i].features[j].entry=entry_chunk.data(buffer_counter); \
			buffer_counter++; \
			\
			if (buffer_counter==elements_in_message) \
			{ \
				read_message(feat_index_chunk); \
				read_message(entry_chunk); \
				buffer_counter=0; \
			} \
		} \
	} \
}

READ_SPARSE_MATRIX(BoolChunk, bool)
READ_SPARSE_MATRIX(Int32Chunk, int8_t)
READ_SPARSE_MATRIX(UInt32Chunk, uint8_t)
READ_SPARSE_MATRIX(UInt32Chunk, char)
READ_SPARSE_MATRIX(Int32Chunk, int32_t)
READ_SPARSE_MATRIX(UInt32Chunk, uint32_t)
READ_SPARSE_MATRIX(Float32Chunk, float32_t)
READ_SPARSE_MATRIX(Float64Chunk, float64_t)
READ_SPARSE_MATRIX(Float64Chunk, floatmax_t)
READ_SPARSE_MATRIX(Int32Chunk, int16_t)
READ_SPARSE_MATRIX(UInt32Chunk, uint16_t)
READ_SPARSE_MATRIX(Int64Chunk, int64_t)
READ_SPARSE_MATRIX(UInt64Chunk, uint64_t)
#undef READ_SPARSE_MATRIX

#define WRITE_SPARSE_MATRIX(chunk_type, sg_type) \
void CProtobufFile::write_sparse_matrix( \
			const SGSparseVector<sg_type>* matrix, int32_t num_vec) \
{ \
	UInt64Chunk feat_index_chunk; \
	chunk_type entry_chunk; \
	int32_t elements_in_message=message_size/sizeof(sg_type); \
	int32_t buffer_counter=0; \
	for (int32_t i=0; i<num_vec; i++) \
	{ \
		for (int32_t j=0; j<matrix[i].num_feat_entries; j++) \
		{ \
			feat_index_chunk.add_data(matrix[i].features[j].feat_index); \
			entry_chunk.add_data(matrix[i].features[j].entry); \
			buffer_counter++; \
			\
			if (buffer_counter==elements_in_message) \
			{ \
				write_message(feat_index_chunk); \
				write_message(entry_chunk); \
				feat_index_chunk.Clear(); \
				entry_chunk.Clear(); \
				buffer_counter=0; \
			} \
		} \
	} \
	\
	if (buffer_counter!=0) \
	{ \
		write_message(feat_index_chunk); \
		write_message(entry_chunk); \
	} \
}

WRITE_SPARSE_MATRIX(BoolChunk, bool)
WRITE_SPARSE_MATRIX(Int32Chunk, int8_t)
WRITE_SPARSE_MATRIX(UInt32Chunk, uint8_t)
WRITE_SPARSE_MATRIX(UInt32Chunk, char)
WRITE_SPARSE_MATRIX(Int32Chunk, int32_t)
WRITE_SPARSE_MATRIX(UInt64Chunk, uint32_t)
WRITE_SPARSE_MATRIX(Int64Chunk, int64_t)
WRITE_SPARSE_MATRIX(UInt64Chunk, uint64_t)
WRITE_SPARSE_MATRIX(Float32Chunk, float32_t)
WRITE_SPARSE_MATRIX(Float64Chunk, float64_t)
WRITE_SPARSE_MATRIX(Float64Chunk, floatmax_t)
WRITE_SPARSE_MATRIX(Int32Chunk, int16_t)
WRITE_SPARSE_MATRIX(UInt32Chunk, uint16_t)
#undef WRITE_SPARSE_MATRIX

#define READ_STRING_LIST(chunk_type, sg_type) \
void CProtobufFile::read_string_list( \
			SGString<sg_type>*& strings, const StringListHeader& data_header) \
{ \
	strings=SG_MALLOC(SGString<sg_type>, data_header.num_str()); \
	\
	chunk_type chunk; \
	read_message(chunk); \
	int32_t elements_in_message=message_size/sizeof(sg_type); \
	int32_t buffer_counter=0; \
	for (uint32_t i=0; i<data_header.num_str(); i++) \
	{ \
		strings[i]=SGString<sg_type>(data_header.str_len(i)); \
		for (int32_t j=0; j<strings[i].slen; j++) \
		{ \
			strings[i].string[j]=chunk.data(buffer_counter); \
			buffer_counter++; \
			\
			if (buffer_counter==elements_in_message) \
			{ \
				read_message(chunk); \
				buffer_counter=0; \
			} \
		} \
	} \
}

READ_STRING_LIST(Int32Chunk, int8_t)
READ_STRING_LIST(UInt32Chunk, uint8_t)
READ_STRING_LIST(UInt32Chunk, char)
READ_STRING_LIST(Int32Chunk, int32_t)
READ_STRING_LIST(UInt32Chunk, uint32_t)
READ_STRING_LIST(Float32Chunk, float32_t)
READ_STRING_LIST(Float64Chunk, float64_t)
READ_STRING_LIST(Float64Chunk, floatmax_t)
READ_STRING_LIST(Int32Chunk, int16_t)
READ_STRING_LIST(UInt32Chunk, uint16_t)
READ_STRING_LIST(Int64Chunk, int64_t)
READ_STRING_LIST(UInt64Chunk, uint64_t)
#undef READ_STRING_LIST

#define WRITE_STRING_LIST(chunk_type, sg_type) \
void CProtobufFile::write_string_list( \
			const SGString<sg_type>* strings, int32_t num_str) \
{ \
	chunk_type chunk; \
	int32_t elements_in_message=message_size/sizeof(sg_type); \
	int32_t buffer_counter=0; \
	for (int32_t i=0; i<num_str; i++) \
	{ \
		for (int32_t j=0; j<strings[i].slen; j++) \
		{ \
			chunk.add_data(strings[i].string[j]); \
			buffer_counter++; \
			\
			if (buffer_counter==elements_in_message) \
			{ \
				write_message(chunk); \
				chunk.Clear(); \
				buffer_counter=0; \
			} \
		} \
	} \
	\
	if (buffer_counter!=0) \
		write_message(chunk); \
}

WRITE_STRING_LIST(Int32Chunk, int8_t)
WRITE_STRING_LIST(UInt32Chunk, uint8_t)
WRITE_STRING_LIST(UInt32Chunk, char)
WRITE_STRING_LIST(Int32Chunk, int32_t)
WRITE_STRING_LIST(UInt64Chunk, uint32_t)
WRITE_STRING_LIST(Int64Chunk, int64_t)
WRITE_STRING_LIST(UInt64Chunk, uint64_t)
WRITE_STRING_LIST(Float32Chunk, float32_t)
WRITE_STRING_LIST(Float64Chunk, float64_t)
WRITE_STRING_LIST(Float64Chunk, floatmax_t)
WRITE_STRING_LIST(Int32Chunk, int16_t)
WRITE_STRING_LIST(UInt32Chunk, uint16_t)
#undef WRITE_STRING_LIST

#endif /* HAVE_PROTOBUF */
