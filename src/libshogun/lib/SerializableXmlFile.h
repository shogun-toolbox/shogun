/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __SERIALIZABLE_XML_FILE_H__
#define __SERIALIZABLE_XML_FILE_H__

#include "lib/config.h"
#ifdef HAVE_XML

#include "lib/SerializableFile.h"
#include "lib/DynamicArray.h"

#include <libxml/parser.h>
#include <libxml/tree.h>

namespace shogun
{
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CSerializableXmlFile
	:public CSerializableFile
{
	CDynamicArray<xmlNode*> m_stack_stream;
	xmlDocPtr m_doc;
	bool m_format;

	void init(const char* fname, bool format);
	bool push_node(const xmlChar* name);
	bool join_node(const xmlChar* name);
	bool next_node(const xmlChar* name);
	void pop_node(void);

protected:
	virtual bool write_scalar_wrapped(
		const TSGDataType* type, const void* param);
	virtual bool read_scalar_wrapped(
		const TSGDataType* type, void* param);

	virtual bool write_cont_begin_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x);
	virtual bool read_cont_begin_wrapped(
		const TSGDataType* type, index_t* len_read_y,
		index_t* len_read_x);

	virtual bool write_cont_end_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x);
	virtual bool read_cont_end_wrapped(
		const TSGDataType* type, index_t len_read_y,
		index_t len_read_x);

	virtual bool write_string_begin_wrapped(
		const TSGDataType* type, index_t length);
	virtual bool read_string_begin_wrapped(
		const TSGDataType* type, index_t* length);

	virtual bool write_string_end_wrapped(
		const TSGDataType* type, index_t length);
	virtual bool read_string_end_wrapped(
		const TSGDataType* type, index_t length);

	virtual bool write_stringentry_begin_wrapped(
		const TSGDataType* type, index_t y);
	virtual bool read_stringentry_begin_wrapped(
		const TSGDataType* type, index_t y);

	virtual bool write_stringentry_end_wrapped(
		const TSGDataType* type, index_t y);
	virtual bool read_stringentry_end_wrapped(
		const TSGDataType* type, index_t y);

	virtual bool write_sparse_begin_wrapped(
		const TSGDataType* type, index_t vec_index,
		index_t length);
	virtual bool read_sparse_begin_wrapped(
		const TSGDataType* type, index_t* vec_index,
		index_t* length);

	virtual bool write_sparse_end_wrapped(
		const TSGDataType* type, index_t vec_index,
		index_t length);
	virtual bool read_sparse_end_wrapped(
		const TSGDataType* type, index_t* vec_index,
		index_t length);

	virtual bool write_sparseentry_begin_wrapped(
		const TSGDataType* type, const TSparseEntry<char>* first_entry,
		index_t feat_index, index_t y);
	virtual bool read_sparseentry_begin_wrapped(
		const TSGDataType* type, TSparseEntry<char>* first_entry,
		index_t* feat_index, index_t y);

	virtual bool write_sparseentry_end_wrapped(
		const TSGDataType* type, const TSparseEntry<char>* first_entry,
		index_t feat_index, index_t y);
	virtual bool read_sparseentry_end_wrapped(
		const TSGDataType* type, TSparseEntry<char>* first_entry,
		index_t* feat_index, index_t y);

	virtual bool write_item_begin_wrapped(
		const TSGDataType* type, index_t y, index_t x);
	virtual bool read_item_begin_wrapped(
		const TSGDataType* type, index_t y, index_t x);

	virtual bool write_item_end_wrapped(
		const TSGDataType* type, index_t y, index_t x);
	virtual bool read_item_end_wrapped(
		const TSGDataType* type, index_t y, index_t x);

	virtual bool write_sgserializable_begin_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic);
	virtual bool read_sgserializable_begin_wrapped(
		const TSGDataType* type, char* sgserializable_name,
		EPrimitiveType* generic);

	virtual bool write_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic);
	virtual bool read_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic);

	virtual bool write_type_begin_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);
	virtual bool read_type_begin_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);

	virtual bool write_type_end_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);
	virtual bool read_type_end_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);

public:
	/** default constructor */
	explicit CSerializableXmlFile(void);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param format indent output; for human readable file
	 */
	explicit CSerializableXmlFile(const char* fname, char rw='r',
								  bool format=false);

	/** default destructor */
	virtual ~CSerializableXmlFile();

	/** @return object name */
	inline virtual const char* get_name() const {
		return "SerializableXmlFile";
	}

	virtual void close(void);
	virtual bool is_opened(void);
};
}
#endif /* HAVE_XML  */
#endif /* __SERIALIZABLE_XML_FILE_H__  */
