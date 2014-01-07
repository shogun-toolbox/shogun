/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <lib/config.h>
#ifdef HAVE_HDF5

#include <io/SerializableHdf5Reader00.h>

using namespace shogun;

SerializableHdf5Reader00::SerializableHdf5Reader00(
	CSerializableHdf5File* file) { m_file = file; }

SerializableHdf5Reader00::~SerializableHdf5Reader00() {}

bool
SerializableHdf5Reader00::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	/* note: param may well be NULL. This doesnt hurt if m->y or m->x are -1 */
	ASSERT(type);

	CSerializableHdf5File::type_item_t* m
		= m_file->m_stack_type.back();

	switch (type->m_stype) {
	case ST_NONE:
		if (m->y != 0 || m->x != 0) return true;
		break;
	case ST_STRING:
		if (m->y == -1 || m->x == -1) break;

		if (m->sub_y != 0) return true;

		ASSERT(param);
		memcpy(param, m->vltype[m->x*m->dims[1] + m->y].p,
			   m->vltype[m->x*m->dims[1] + m->y].len
			   *type->sizeof_ptype());

		return true;
	case ST_SPARSE:
		if (m->sub_y != 0) return true;
		break;
	case ST_UNDEFINED:
		return false;
	}

	hid_t mem_type_id;
	if ((mem_type_id = CSerializableHdf5File::new_stype2hdf5(
			 type->m_stype, type->m_ptype)) < 0) return false;

	switch (type->m_stype) {
	case ST_NONE:
		if (H5Dread(m->dset, mem_type_id, H5S_ALL, H5S_ALL,
					H5P_DEFAULT, param) < 0) return false;
		break;
	case ST_STRING:
		if (H5Dread(m->dset, mem_type_id, H5S_ALL, H5S_ALL,
					H5P_DEFAULT, m->vltype) < 0) return false;
		break;
	case ST_SPARSE:
		if (H5Dread(m->dset, m->dtype, H5S_ALL, H5S_ALL,
					H5P_DEFAULT, m->sparse_ptr) < 0) return false;
		break;
	case ST_UNDEFINED:
		return false;
	}

	if (H5Tclose(mem_type_id) < 0) return false;

	return true;
}

bool
SerializableHdf5Reader00::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	CSerializableHdf5File::type_item_t* m
		= m_file->m_stack_type.back();

	if (type->m_ptype != PT_SGOBJECT) {
		switch (type->m_ctype) {
		case CT_NDARRAY:
			SG_NOTIMPLEMENTED
		case CT_SCALAR:
			SG_ERROR("read_cont_begin_wrapped(): Implementation error"
					 " during writing Hdf5File (0)!");
			return false;
		case CT_VECTOR: case CT_SGVECTOR: *len_read_y = m->dims[0]; break;
		case CT_MATRIX: case CT_SGMATRIX:
			*len_read_x = m->dims[0]; *len_read_y = m->dims[1];
			break;
		default: return false;
		}

		return true;
	}

	if (!m_file->attr_exists(STR_IS_CONT)) return false;

	string_t ctype_buf, buf;
	type->to_string(ctype_buf, STRING_LEN);
	if (!m_file->attr_read_string(STR_CTYPE_NAME, buf, STRING_LEN))
		return false;
	if (strcmp(ctype_buf, buf) != 0) return false;

	switch (type->m_ctype) {
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
	case CT_SCALAR:
		SG_ERROR("read_cont_begin_wrapped(): Implementation error"
				 " during writing Hdf5File (1)!");
		return false;
	case CT_MATRIX: case CT_SGMATRIX:
		if (!m_file->attr_read_scalar(TYPE_INDEX, STR_LENGTH_X,
									  len_read_x))
			return false;
		/* break;  */
	case CT_VECTOR: case CT_SGVECTOR:
		if (!m_file->attr_read_scalar(TYPE_INDEX, STR_LENGTH_Y,
									  len_read_y))
			return false;
		break;
	default: return false;
	}

	return true;
}

bool
SerializableHdf5Reader00::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	return true;
}

bool
SerializableHdf5Reader00::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	CSerializableHdf5File::type_item_t* m
		= m_file->m_stack_type.back();

	if (m->y == 0 && m->x == 0) {
		m->y = -1; m->x = -1;
		read_scalar_wrapped(type, NULL);
		m->y = 0; m->x = 0;
	}

	*length = m->vltype[m->x*m->dims[1] + m->y].len;

	return true;
}

bool
SerializableHdf5Reader00::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
SerializableHdf5Reader00::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	CSerializableHdf5File::type_item_t* m
		= m_file->m_stack_type.back();

	m->sub_y = y;

	return true;
}

bool
SerializableHdf5Reader00::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
SerializableHdf5Reader00::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	CSerializableHdf5File::type_item_t* m_prev
		= m_file->m_stack_type.back();

	if(!m_file->dspace_select(type->m_ctype, m_prev->y, m_prev->x))
		return false;

	CSerializableHdf5File::type_item_t* m = new CSerializableHdf5File
		::type_item_t(m_prev->name);
	m_file->m_stack_type.push_back(m);

	/* ************************************************************ */

	if (!m_file->group_open(m->name, STR_GROUP_PREFIX)) return false;
	if (!m_file->attr_exists(STR_IS_SPARSE)) return false;

	string_t name;
	CSerializableHdf5File::index2string(
		name, STRING_LEN, type->m_ctype, m_prev->y, m_prev->x);
	if ((m->dset = H5Dopen2(m_file->m_stack_h5stream.back(), name,
							H5P_DEFAULT)) < 0)
		return false;

	if ((m->dtype = H5Dget_type(m->dset)) < 0) return false;
	if (!CSerializableHdf5File::isequal_stype2hdf5(
			type->m_stype, type->m_ptype, m->dtype)) return false;

	if ((m->dspace = H5Dget_space(m->dset)) < 0) return false;
	if (H5Sget_simple_extent_ndims(m->dspace) != 1) return false;


	if ((m->rank = H5Sget_simple_extent_dims(m->dspace, m->dims, NULL)
			) < 0) return false;

	if (H5Sget_simple_extent_type(m->dspace) != H5S_NULL
		&& m->rank != 1) return false;

	*length = m->dims[0];

	/* ************************************************************ */

	char* buf = SG_MALLOC(char, CSerializableHdf5File::sizeof_sparsetype());

	hid_t mem_type_id;
	if ((mem_type_id = CSerializableHdf5File::new_sparsetype()) < 0)
		return false;

	hid_t mem_space_id;
	if ((mem_space_id = H5Screate_simple(0, NULL, NULL)) < 0)
		return false;

	if (H5Dread(m_prev->dset, mem_type_id, mem_space_id,
				m_prev->dspace, H5P_DEFAULT, buf) < 0) return false;

	if (H5Sclose(mem_space_id) < 0) return false;
	if (H5Tclose(mem_type_id) < 0) return false;

	delete buf;

	return true;
}

bool
SerializableHdf5Reader00::read_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (!m_file->group_close()) return false;

	delete m_file->m_stack_type.back();
	m_file->m_stack_type.pop_back();

	return true;
}

bool
SerializableHdf5Reader00::read_sparseentry_begin_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	CSerializableHdf5File::type_item_t* m
		= m_file->m_stack_type.back();

	m->sparse_ptr = first_entry;
	m->sub_y = y;

	return true;
}

bool
SerializableHdf5Reader00::read_sparseentry_end_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	return true;
}

bool
SerializableHdf5Reader00::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	CSerializableHdf5File::type_item_t* m
		= m_file->m_stack_type.back();
	m->y = y; m->x = x;

	if (type->m_ptype != PT_SGOBJECT) return true;

	string_t name;
	if (!CSerializableHdf5File::index2string(
			name, STRING_LEN, type->m_ctype, y, x)) return false;
	if (!m_file->group_open(name, "")) return false;

	return true;
}

bool
SerializableHdf5Reader00::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (type->m_ptype == PT_SGOBJECT)
		if (!m_file->group_close()) return false;

	return true;
}

bool
SerializableHdf5Reader00::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	if (!m_file->attr_exists(STR_IS_SGSERIALIZABLE)) return false;

	if (m_file->attr_exists(STR_IS_NULL)) {
		*sgserializable_name = '\0'; return true;
	}

	if (!m_file->attr_read_string(
			STR_INSTANCE_NAME, sgserializable_name, STRING_LEN))
		return false;

	if (m_file->attr_exists(STR_GENERIC_NAME)) {
		string_t buf;
		if (!m_file->attr_read_string(
				STR_GENERIC_NAME, buf, STRING_LEN)) return false;
		if (!TSGDataType::string_to_ptype(generic, buf))
			return false;
	}

	return true;
}

bool
SerializableHdf5Reader00::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
SerializableHdf5Reader00::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	CSerializableHdf5File::type_item_t* m = new CSerializableHdf5File
		::type_item_t(name);
	m_file->m_stack_type.push_back(m);

	if (type->m_ptype == PT_SGOBJECT) {
		if (!m_file->group_open(name, "")) return false;
		return true;
	}

	if ((m->dset = H5Dopen2(m_file->m_stack_h5stream.back(), name,
							H5P_DEFAULT)) < 0)
		return false;

	if ((m->dtype = H5Dget_type(m->dset)) < 0) return false;
	if (!CSerializableHdf5File::isequal_stype2hdf5(
			type->m_stype, type->m_ptype, m->dtype)) return false;

	if ((m->dspace = H5Dget_space(m->dset)) < 0) return false;

	if (H5Sget_simple_extent_ndims(m->dspace) > 2) return false;
	if ((m->rank = H5Sget_simple_extent_dims(m->dspace, m->dims, NULL)
			) < 0) return false;

	switch (type->m_ctype) {
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
	case CT_SCALAR:
		if (m->rank != 0) return false;
		if (type->m_stype == ST_STRING) m->vltype = SG_MALLOC(hvl_t, 1);
		break;
	case CT_VECTOR: case CT_SGVECTOR:
		if (H5Sget_simple_extent_type(m->dspace) != H5S_NULL
			&& m->rank != 1) return false;
		if (type->m_stype == ST_STRING)
			m->vltype = SG_MALLOC(hvl_t, m->dims[0]);
		break;
	case CT_MATRIX: case CT_SGMATRIX:
		if (H5Sget_simple_extent_type(m->dspace) != H5S_NULL
			&& m->rank != 2) return false;
		if (type->m_stype == ST_STRING)
			m->vltype = SG_MALLOC(hvl_t, m->dims[0] *m->dims[1]);
		break;
	default: return false;
	}

	return true;
}

bool
SerializableHdf5Reader00::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (type->m_ptype == PT_SGOBJECT)
		if (!m_file->group_close()) return false;

	delete m_file->m_stack_type.back();
	m_file->m_stack_type.pop_back();
	return true;
}

#endif /* HAVE_HDF5  */
