/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __SERIALIZABLE_ASCII_FILE_H__
#define __SERIALIZABLE_ASCII_FILE_H__

#include <shogun/lib/config.h>
#include <shogun/io/SerializableFile.h>
#include <shogun/base/DynArray.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/common.h>

#define CHAR_CONT_BEGIN            '('
#define CHAR_CONT_END              ')'
#define CHAR_ITEM_BEGIN            '{'
#define CHAR_ITEM_END              '}'
#define CHAR_SGSERIAL_BEGIN        '['
#define CHAR_SGSERIAL_END          ']'
#define CHAR_STRING_BEGIN          CHAR_SGSERIAL_BEGIN
#define CHAR_STRING_END            CHAR_SGSERIAL_END
#define CHAR_SPARSE_BEGIN          CHAR_CONT_BEGIN
#define CHAR_SPARSE_END            CHAR_CONT_END

#define CHAR_TYPE_END              '\n'

#define STR_SGSERIAL_NULL          "null"

namespace shogun
{
template <class T> struct SGSparseVectorEntry;

/** @brief serializable ascii file */
class CSerializableAsciiFile :public CSerializableFile
{
	friend class SerializableAsciiReader00;

	DynArray<long> m_stack_fpos;

	void init();
	bool ignore();

protected:

	/** new reader
	 * @param dest_version
	 * @param n
	 */
	virtual TSerializableReader* new_reader(
		char* dest_version, size_t n);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	virtual bool write_scalar_wrapped(
		const TSGDataType* type, const void* param);

	virtual bool write_cont_begin_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x);
	virtual bool write_cont_end_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x);

	virtual bool write_string_begin_wrapped(
		const TSGDataType* type, index_t length);
	virtual bool write_string_end_wrapped(
		const TSGDataType* type, index_t length);

	virtual bool write_stringentry_begin_wrapped(
		const TSGDataType* type, index_t y);
	virtual bool write_stringentry_end_wrapped(
		const TSGDataType* type, index_t y);

	virtual bool write_sparse_begin_wrapped(
		const TSGDataType* type, index_t length);
	virtual bool write_sparse_end_wrapped(
		const TSGDataType* type, index_t length);

	virtual bool write_sparseentry_begin_wrapped(
		const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
		index_t feat_index, index_t y);
	virtual bool write_sparseentry_end_wrapped(
		const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
		index_t feat_index, index_t y);

	virtual bool write_item_begin_wrapped(
		const TSGDataType* type, index_t y, index_t x);
	virtual bool write_item_end_wrapped(
		const TSGDataType* type, index_t y, index_t x);

	virtual bool write_sgserializable_begin_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic);
	virtual bool write_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic);

	virtual bool write_type_begin_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);
	virtual bool write_type_end_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);
#endif
public:
	/** default constructor */
	explicit CSerializableAsciiFile();

	/** constructor
	 *
	 * @param fstream already opened file
	 * @param rw
	 */
	explicit CSerializableAsciiFile(FILE* fstream, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerializableAsciiFile(const char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerializableAsciiFile();

	/** @return object name */
	virtual const char* get_name() const {
		return "SerializableAsciiFile";
	}
};
}

#endif /* __SERIALIZABLE_ASCII_FILE_H__  */
