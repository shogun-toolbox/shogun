/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef __SERIALIZABLE_FILE_H__
#define __SERIALIZABLE_FILE_H__

#include <stdio.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>

namespace shogun
{
	class CSGObject;

class CSerializableFile :public CSGObject
{
public:
	struct TSerializableReader :public CSGObject {

		/* ******************************************************** */
		/* Begin of abstract write methods  */

		virtual bool read_scalar_wrapped(
			const TSGDataType* type, void* param) = 0;

		virtual bool read_cont_begin_wrapped(
			const TSGDataType* type, index_t* len_read_y,
			index_t* len_read_x) = 0;
		virtual bool read_cont_end_wrapped(
			const TSGDataType* type, index_t len_read_y,
			index_t len_read_x) = 0;

		virtual bool read_string_begin_wrapped(
			const TSGDataType* type, index_t* length) = 0;
		virtual bool read_string_end_wrapped(
			const TSGDataType* type, index_t length) = 0;

		virtual bool read_stringentry_begin_wrapped(
			const TSGDataType* type, index_t y) = 0;
		virtual bool read_stringentry_end_wrapped(
			const TSGDataType* type, index_t y) = 0;

		virtual bool read_sparse_begin_wrapped(
			const TSGDataType* type, index_t* vec_index,
			index_t* length) = 0;
		virtual bool read_sparse_end_wrapped(
			const TSGDataType* type, index_t* vec_index,
			index_t length) = 0;

		virtual bool read_sparseentry_begin_wrapped(
			const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
			index_t* feat_index, index_t y) = 0;
		virtual bool read_sparseentry_end_wrapped(
			const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
			index_t* feat_index, index_t y) = 0;

		virtual bool read_item_begin_wrapped(
			const TSGDataType* type, index_t y, index_t x) = 0;
		virtual bool read_item_end_wrapped(
			const TSGDataType* type, index_t y, index_t x) = 0;

		virtual bool read_sgserializable_begin_wrapped(
			const TSGDataType* type, char* sgserializable_name,
			EPrimitiveType* generic) = 0;
		virtual bool read_sgserializable_end_wrapped(
			const TSGDataType* type, const char* sgserializable_name,
			EPrimitiveType generic) = 0;

		virtual bool read_type_begin_wrapped(
			const TSGDataType* type, const char* name,
			const char* prefix) = 0;
		virtual bool read_type_end_wrapped(
			const TSGDataType* type, const char* name,
			const char* prefix) = 0;

		/* End of abstract write methods  */
		/* ******************************************************** */

	}; /* struct TSerializableReader  */
/* public:  */
private:
	TSerializableReader* m_reader;

	bool is_task_warn(char rw, const char* name, const char* prefix);
	bool false_warn(const char* prefix, const char* name);

protected:
	FILE* m_fstream;
	char m_task;
	char* m_filename;

	virtual void init(FILE* fstream, char task, const char* filename);

	/* ************************************************************ */
	/* Begin of abstract write methods  */

	virtual TSerializableReader* new_reader(
		char* dest_version, size_t n) = 0;

	virtual bool write_scalar_wrapped(
		const TSGDataType* type, const void* param) = 0;

	virtual bool write_cont_begin_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x) = 0;
	virtual bool write_cont_end_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x) = 0;

	virtual bool write_string_begin_wrapped(
		const TSGDataType* type, index_t length) = 0;
	virtual bool write_string_end_wrapped(
		const TSGDataType* type, index_t length) = 0;

	virtual bool write_stringentry_begin_wrapped(
		const TSGDataType* type, index_t y) = 0;
	virtual bool write_stringentry_end_wrapped(
		const TSGDataType* type, index_t y) = 0;

	virtual bool write_sparse_begin_wrapped(
		const TSGDataType* type, index_t vec_index,
		index_t length) = 0;
	virtual bool write_sparse_end_wrapped(
		const TSGDataType* type, index_t vec_index,
		index_t length) = 0;

	virtual bool write_sparseentry_begin_wrapped(
		const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
		index_t feat_index, index_t y) = 0;
	virtual bool write_sparseentry_end_wrapped(
		const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
		index_t feat_index, index_t y) = 0;

	virtual bool write_item_begin_wrapped(
		const TSGDataType* type, index_t y, index_t x) = 0;
	virtual bool write_item_end_wrapped(
		const TSGDataType* type, index_t y, index_t x) = 0;

	virtual bool write_sgserializable_begin_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic) = 0;
	virtual bool write_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic) = 0;

	virtual bool write_type_begin_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix) = 0;
	virtual bool write_type_end_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix) = 0;

	/* End of abstract write methods  */
	/* ************************************************************ */

public:
	/** default constructor */
	explicit CSerializableFile(void);

	/** constructor
	 *
	 * @param f already opened file
	 */
	explicit CSerializableFile(FILE* fstream, char rw);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerializableFile(const char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerializableFile(void);

	/** close */
	virtual void close(void);

	/** is opened */
	virtual bool is_opened(void);

	/* ************************************************************ */
	/* Begin of public wrappers  */

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	virtual bool write_scalar(
		const TSGDataType* type, const char* name, const char* prefix,
		const void* param);
	virtual bool read_scalar(
		const TSGDataType* type, const char* name, const char* prefix,
		void* param);

	virtual bool write_cont_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t len_real_y, index_t len_real_x);
	virtual bool read_cont_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t* len_read_y, index_t* len_read_x);

	virtual bool write_cont_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t len_real_y, index_t len_real_x);
	virtual bool read_cont_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t len_read_y, index_t len_read_x);

	virtual bool write_string_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t length);
	virtual bool read_string_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t* length);

	virtual bool write_string_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t length);
	virtual bool read_string_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t length);

	virtual bool write_stringentry_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y);
	virtual bool read_stringentry_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y);

	virtual bool write_stringentry_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y);
	virtual bool read_stringentry_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y);

	virtual bool write_sparse_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t vec_index, index_t length);
	virtual bool read_sparse_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t* vec_index, index_t* length);

	virtual bool write_sparse_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t vec_index, index_t length);
	virtual bool read_sparse_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t* vec_index, index_t length);

	virtual bool write_sparseentry_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		const SGSparseVectorEntry<char>* first_entry, index_t feat_index,
		index_t y);
	virtual bool read_sparseentry_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		SGSparseVectorEntry<char>* first_entry, index_t* feat_index,
		index_t y);

	virtual bool write_sparseentry_end(
		const TSGDataType* type, const char* name, const char* prefix,
		const SGSparseVectorEntry<char>* first_entry, index_t feat_index,
		index_t y);
	virtual bool read_sparseentry_end(
		const TSGDataType* type, const char* name, const char* prefix,
		SGSparseVectorEntry<char>* first_entry, index_t* feat_index,
		index_t y);

	virtual bool write_item_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);
	virtual bool read_item_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);

	virtual bool write_item_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);
	virtual bool read_item_end(
		const TSGDataType* type, const char* name, const char* prefix,
		index_t y, index_t x);

	virtual bool write_sgserializable_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		const char* sgserializable_name, EPrimitiveType generic);
	virtual bool read_sgserializable_begin(
		const TSGDataType* type, const char* name, const char* prefix,
		char* sgserializable_name, EPrimitiveType* generic);

	virtual bool write_sgserializable_end(
		const TSGDataType* type, const char* name, const char* prefix,
		const char* sgserializable_name, EPrimitiveType generic);
	virtual bool read_sgserializable_end(
		const TSGDataType* type, const char* name, const char* prefix,
		const char* sgserializable_name, EPrimitiveType generic);

	virtual bool write_type_begin(
		const TSGDataType* type, const char* name, const char* prefix);
	virtual bool read_type_begin(
		const TSGDataType* type, const char* name, const char* prefix);

	virtual bool write_type_end(
		const TSGDataType* type, const char* name, const char* prefix);
	virtual bool read_type_end(
		const TSGDataType* type, const char* name, const char* prefix);
#endif
	/* End of public wrappers  */
	/* ************************************************************ */
};
}
#endif // __SERIALIZABLE_FILE_H__
