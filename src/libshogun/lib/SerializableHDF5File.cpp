/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/config.h"
#ifdef HAVE_HDF5

#include "lib/SerializableHDF5File.h"

#define NOT_OPEN                   ((hid_t) -1)

#define TYPE_INDEX                 H5T_NATIVE_INT32

#define STR_IS_SGSERIALIZABLE      "is_sgserializable"
#define STR_IS_SPARSE              "is_sparse"
#define STR_IS_CONT                "is_container"
#define STR_IS_NULL                "is_null"
#define STR_INSTANCE_NAME          "instance_name"
#define STR_GENERIC_NAME           "generic_name"
#define STR_CTYPE_NAME             "container_type"
#define STR_LENGTH_X               "length_x"
#define STR_LENGTH_Y               "length_y"

#define STR_GROUP_PREFIX           "$"

#define STR_SPARSE_VINDEX          "vec_index"
#define STR_SPARSE_FPTR            "features_ptr"
#define STR_SPARSEENTRY_FINDEX     "feat_index"
#define STR_SPARSEENTRY_ENTRY      "entry"

using namespace shogun;

CSerializableHDF5File::type_item_t::type_item_t(const char* name_)
{
	rank = 0;
	dims[0] = dims[1] = 0;
	dspace = dtype = dset = NOT_OPEN;
	y = x = 0;
	name = name_;
	sparse_buf.feat_index = 0;
}

CSerializableHDF5File::type_item_t::~type_item_t(void)
{
	if (dset >= 0) H5Dclose(dset);
	if (dtype >= 0) H5Tclose(dtype);
	if (dspace >= 0) H5Sclose(dspace);
}

hid_t
CSerializableHDF5File::sizeof_sparsetype(void) {
	return H5Tget_size(TYPE_INDEX) + H5Tget_size(H5T_STD_REF_OBJ);
}
hid_t
CSerializableHDF5File::new_sparsetype(void)
{
	hid_t result = H5Tcreate(H5T_COMPOUND, sizeof_sparsetype());

	if (H5Tinsert(result, STR_SPARSE_VINDEX, 0, TYPE_INDEX) < 0)
		return NOT_OPEN;
	if (H5Tinsert(result, STR_SPARSE_FPTR, H5Tget_size(TYPE_INDEX),
				  H5T_STD_REF_OBJ) < 0)
		return NOT_OPEN;

	return result;
}
hobj_ref_t*
CSerializableHDF5File::get_ref_sparstype(void* sparse_buf) {
	return (hobj_ref_t*)
		((char*) sparse_buf + H5Tget_size(TYPE_INDEX));
}

hid_t
CSerializableHDF5File::new_sparseentrytype(EPrimitiveType ptype)
{
	hid_t result = H5Tcreate(H5T_COMPOUND,
							 TSGDataType::sizeof_sparseentry(ptype));
	if (result < 0) return NOT_OPEN;

	if (H5Tinsert(result, STR_SPARSEENTRY_FINDEX,
				  HOFFSET(TSparseEntry<char>, feat_index), TYPE_INDEX)
		< 0) return NOT_OPEN;
	if (H5Tinsert(result, STR_SPARSEENTRY_ENTRY,
				  HOFFSET(TSparseEntry<char>, entry),
				  ptype2hdf5(ptype)) < 0) return NOT_OPEN;

	return result;
}


hid_t
CSerializableHDF5File::ptype2hdf5(EPrimitiveType ptype)
{
	switch (ptype) {
	case PT_BOOL: return H5T_NATIVE_HBOOL; break;
	case PT_CHAR: return H5T_NATIVE_CHAR; break;
	case PT_INT8: return H5T_NATIVE_INT8; break;
	case PT_UINT8: return H5T_NATIVE_UINT8; break;
	case PT_INT16: return H5T_NATIVE_INT16; break;
	case PT_UINT16: return H5T_NATIVE_UINT16; break;
	case PT_INT32: return H5T_NATIVE_INT32; break;
	case PT_UINT32: return H5T_NATIVE_UINT32; break;
	case PT_INT64: return H5T_NATIVE_INT64; break;
	case PT_UINT64: return H5T_NATIVE_UINT64; break;
	case PT_FLOAT32: return H5T_NATIVE_FLOAT; break;
	case PT_FLOAT64: return H5T_NATIVE_DOUBLE; break;
	case PT_FLOATMAX: return H5T_NATIVE_LDOUBLE; break;
	case PT_SGSERIALIZABLE_PTR: return NOT_OPEN; break;
	}

	return NOT_OPEN;
}

hid_t
CSerializableHDF5File::new_stype2hdf5(EStructType stype,
									  EPrimitiveType ptype)
{
	hid_t result = ptype2hdf5(ptype);

	switch (stype) {
	case ST_NONE: result = H5Tcopy(result); break;
	case ST_STRING: result = H5Tvlen_create(result); break;
	case ST_SPARSE: result = new_sparsetype(); break;
	}

	return result;
}

bool
CSerializableHDF5File::index2string(
	char* dest, size_t n, EContainerType ctype, index_t y, index_t x)
{
	switch (ctype) {
	case CT_SCALAR: return false;
	case CT_VECTOR: snprintf(dest, n, "y%u", y); break;
	case CT_MATRIX: snprintf(dest, n, "y%u_x%u", y, x); break;
	}

	return true;
}

bool
CSerializableHDF5File::isequal_stype2hdf5(EStructType stype,
										  EPrimitiveType ptype,
										  hid_t htype)
{
	hid_t pbuf = ptype2hdf5(ptype), pbuf2 = NOT_OPEN;

	bool to_close = false;
	switch (stype) {
	case ST_NONE: break;
	case ST_STRING:
		to_close = true; pbuf = H5Tvlen_create(pbuf); break;
	case ST_SPARSE:
		to_close = true; pbuf = new_sparsetype();
		pbuf2 = new_sparseentrytype(ptype); break;
	}

	bool result = (H5Tequal(htype, pbuf) > 0)
		|| (pbuf2 >= 0 && H5Tequal(htype, pbuf2) > 0);

	if (pbuf2 >= 0 && H5Tclose(pbuf2) < 0) return false;
	if (to_close && H5Tclose(pbuf) < 0) return false;
	return result;
}

bool
CSerializableHDF5File::dspace_select(EContainerType ctype, index_t y,
									 index_t x)
{
	type_item_t* m = m_stack_type.back();

	if (H5Sselect_none(m->dspace) < 0) return false;

	hsize_t coord[2];
	switch (ctype) {
	case CT_SCALAR: return false;
	case CT_MATRIX: coord[1] = x; /* break;  */
	case CT_VECTOR: coord[0] = y; break;
	}
	if (H5Sselect_elements(m->dspace, H5S_SELECT_SET, 1, coord) < 0)
		return false;

	return true;
}

bool
CSerializableHDF5File::attr_write_scalar(
	hid_t datatype, const char* name, const void* val)
{
	hid_t dspace;
	if ((dspace = H5Screate_simple(0, NULL, NULL)) < 0) return false;
	hid_t dtype;
	if ((dtype = H5Tcopy(datatype)) < 0) return false;
	hid_t attr;
	if ((attr = H5Acreate2(
			 m_stack_h5stream.back(), name, dtype, dspace,
			 H5P_DEFAULT, H5P_DEFAULT)) < 0) return false;

	if (H5Awrite(attr, datatype, val) < 0) return false;

	if (H5Aclose(attr) < 0) return false;
	if (H5Tclose(dtype) < 0) return false;
	if (H5Sclose(dspace) < 0) return false;

	return true;
}

bool
CSerializableHDF5File::attr_write_string(
	const char* name, const char* val)
{
	hid_t dtype;
	if ((dtype = H5Tcopy(H5T_C_S1)) < 0) return false;
	if (H5Tset_size(dtype, strlen(val)+1) < 0) return false;

	if (!attr_write_scalar(dtype, name, val)) return false;

	if (H5Tclose(dtype) < 0) return false;

	return true;
}

bool
CSerializableHDF5File::attr_exists(const char* name)
{
	return H5Aexists(m_stack_h5stream.back(), name) > 0;
}

size_t
CSerializableHDF5File::attr_get_size(const char* name)
{
	if (!attr_exists(name)) return 0;

	hid_t attr;
	if ((attr = H5Aopen(m_stack_h5stream.back(), name, H5P_DEFAULT))
		< 0) return 0;

	hid_t dtype;
	if ((dtype = H5Aget_type(attr)) < 0) return 0;

	size_t result = H5Tget_size(dtype);

	if (H5Tclose(dtype) < 0) return 0;
	if (H5Aclose(attr) < 0) return 0;

	return result;
}

bool
CSerializableHDF5File::attr_read_scalar(
	hid_t datatype, const char* name, void* val)
{
	if (!attr_exists(name)) return false;

	hid_t attr;
	if ((attr = H5Aopen(m_stack_h5stream.back(), name, H5P_DEFAULT))
		< 0) return false;

	hid_t dspace;
	if ((dspace = H5Aget_space(attr)) < 0) return false;
	if (H5Sget_simple_extent_type(dspace) != H5S_SCALAR) return false;

	hid_t dtype;
	if ((dtype = H5Aget_type(attr)) < 0) return false;
	if (H5Tequal(datatype, dtype) <= 0) return false;

	if (H5Aread(attr, datatype, val) < 0) return false;

	if (H5Tclose(dtype) < 0) return false;
	if (H5Sclose(dspace) < 0) return false;
	if (H5Aclose(attr) < 0) return false;

	return true;
}

bool
CSerializableHDF5File::attr_read_string(
	const char* name, char* val, size_t n)
{
	size_t size = attr_get_size(name);
	if (size == 0 || size > n) return false;

	hid_t dtype;
	if ((dtype = H5Tcopy(H5T_C_S1)) < 0) return false;
	if (H5Tset_size(dtype, size) < 0) return false;

	if (!attr_read_scalar(dtype, name, val)) return false;

	if (H5Tclose(dtype) < 0) return false;

	return true;
}

bool
CSerializableHDF5File::group_create(const char* name,
									const char* prefix)
{
	hid_t ngroup;
	string_t gname;

	snprintf(gname, STRING_LEN, "%s%s", prefix, name);

	m_stack_h5stream.push_back(
		ngroup = H5Gcreate2(m_stack_h5stream.back(), gname, H5P_DEFAULT,
							H5P_DEFAULT, H5P_DEFAULT));
	if (ngroup < 0) return false;

	return true;
}

bool
CSerializableHDF5File::group_open(const char* name,
								  const char* prefix)
{
	hid_t group;
	string_t gname;

	snprintf(gname, STRING_LEN, "%s%s", prefix, name);

	m_stack_h5stream.push_back(
		group = H5Gopen2(m_stack_h5stream.back(), gname, H5P_DEFAULT));
	if (group < 0) return false;

	return true;
}

bool
CSerializableHDF5File::group_close(void)
{
	if (H5Gclose(m_stack_h5stream.back()) < 0) return false;
	m_stack_h5stream.pop_back();

	return true;
}

CSerializableHDF5File::CSerializableHDF5File(void)
	:CSerializableFile() { init(""); }

CSerializableHDF5File::CSerializableHDF5File(const char* fname, char rw)
	:CSerializableFile()
{
	CSerializableFile::init(NULL, rw, fname);
	init(fname);
}

CSerializableHDF5File::~CSerializableHDF5File()
{
	while (m_stack_type.get_num_elements() > 0) {
		delete m_stack_type.back(); m_stack_type.pop_back();
	}

	close();
}

void
CSerializableHDF5File::init(const char* fname)
{
	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n");
		close(); return;
	}

	hid_t h5stream;
	switch (m_task) {
	case 'w':
		h5stream = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT,
							 H5P_DEFAULT);
		break;
	case 'r':
		h5stream = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
		break;
	default:
		break;
	}

	if (h5stream < 0) {
		SG_WARNING("Could not open file `%s'!\n", m_filename);
		close(); return;
	}

	m_stack_h5stream.push_back(h5stream);
}

void
CSerializableHDF5File::close(void)
{
	while (m_stack_h5stream.get_num_elements() > 1) {
		if (m_stack_h5stream.back() >= 0)
			H5Gclose(m_stack_h5stream.back());
		m_stack_h5stream.pop_back();
	}

	if (m_stack_h5stream.get_num_elements() == 1) {
		if (m_stack_h5stream.back() >= 0)
			H5Fclose(m_stack_h5stream.back());
		m_stack_h5stream.pop_back();
	}
}

bool
CSerializableHDF5File::is_opened(void)
{
	return m_stack_h5stream.get_num_elements() > 0;
}

bool
CSerializableHDF5File::write_scalar_wrapped(
	const TSGDataType* type, const void* param)
{
	type_item_t* m = m_stack_type.back();

	if (type->m_ptype == PT_SGSERIALIZABLE_PTR) {
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
				 " writing HDF5File!");
		return false;
	}

	hid_t mem_type_id;
	if ((mem_type_id = new_stype2hdf5(type->m_stype, type->m_ptype)
			) < 0)
		return false;

	hid_t mem_space_id;
	if ((mem_space_id = H5Screate_simple(0, NULL, NULL)) < 0)
		return false;

	switch (type->m_stype) {
	case ST_NONE:
		hbool_t buf;
		if (type->m_ptype == PT_BOOL) {
			buf = (hbool_t) *(bool*) param;
			param = &buf;
		}

		if (H5Dwrite(m->dset, mem_type_id, mem_space_id, m->dspace,
					 H5P_DEFAULT, param) < 0) return false;
		break;
	case ST_STRING:
		m->vltype.p = (void*) param;
		if (m->vltype.len > 0
			&& H5Dwrite(m->dset, mem_type_id, mem_space_id, m->dspace,
						H5P_DEFAULT, &m->vltype) < 0) return false;
		break;
	case ST_SPARSE:
		memcpy(&m->sparse_buf.entry, param, type->sizeof_ptype());
		if (H5Dwrite(m->dset, m->dtype, mem_space_id, m->dspace,
					 H5P_DEFAULT, &m->sparse_buf) < 0) return false;
		break;
	}

	if (H5Sclose(mem_space_id) < 0) return false;
	if (H5Tclose(mem_type_id) < 0) return false;

	return true;
}

bool
CSerializableHDF5File::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	type_item_t* m = m_stack_type.back();

	if (type->m_ptype == PT_SGSERIALIZABLE_PTR) {
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
				 " writing HDF5File!");
		return false;
	}

	hid_t mem_type_id;
	if ((mem_type_id = new_stype2hdf5(type->m_stype, type->m_ptype)) < 0)
		return false;

	hid_t mem_space_id;
	if ((mem_space_id = H5Screate_simple(0, NULL, NULL)) < 0)
		return false;

	size_t bytes = 0;
	switch (type->m_stype) {
	case ST_NONE:
		if (type->m_ptype == PT_BOOL) {
			hbool_t buf;
			if (H5Dread(m->dset, mem_type_id, mem_space_id, m->dspace,
						H5P_DEFAULT, &buf) < 0) return false;
			*(bool*) param = (bool) buf;
		} else
			if (H5Dread(m->dset, mem_type_id, mem_space_id, m->dspace,
						H5P_DEFAULT, param) < 0) return false;
		break;
	case ST_STRING:
		bytes = m->vltype.len;
		if (m->vltype.len > 0
			&& H5Dread(m->dset, mem_type_id, mem_space_id, m->dspace,
					   H5P_DEFAULT, &m->vltype) < 0) return false;
		memcpy(param, m->vltype.p, bytes);
		if (H5Dvlen_reclaim (mem_type_id, mem_space_id, H5P_DEFAULT,
							 &m->vltype) < 0) return false;
		break;
	case ST_SPARSE:
		if (H5Dread(m->dset, m->dtype, mem_space_id, m->dspace,
					H5P_DEFAULT, &m->sparse_buf) < 0) return false;
		memcpy(param, &m->sparse_buf.entry, type->sizeof_ptype());
		break;
	}

	if (H5Sclose(mem_space_id) < 0) return false;
	if (H5Tclose(mem_type_id) < 0) return false;

	return true;
}

bool
CSerializableHDF5File::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	hbool_t bool_buf = true;

	if (type->m_ptype != PT_SGSERIALIZABLE_PTR) return true;

	if (!attr_write_scalar(H5T_NATIVE_HBOOL, STR_IS_CONT, &bool_buf))
		return false;

	string_t ctype_buf;
	type->to_string(ctype_buf, STRING_LEN);
	if (!attr_write_string(STR_CTYPE_NAME, ctype_buf)) return false;

	switch (type->m_ctype) {
	case CT_SCALAR:
		SG_ERROR("write_cont_begin_wrapped(): Implementation error "
				 "during writing HDF5File!");
		return false;
	case CT_MATRIX:
		if (!attr_write_scalar(TYPE_INDEX, STR_LENGTH_X, &len_real_x))
			return false;
		/* break;  */
	case CT_VECTOR:
		if (!attr_write_scalar(TYPE_INDEX, STR_LENGTH_Y, &len_real_y))
			return false;
		break;
	}

	return true;
}

bool
CSerializableHDF5File::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	type_item_t* m = m_stack_type.back();

	if (type->m_ptype != PT_SGSERIALIZABLE_PTR) {
		switch (type->m_ctype) {
		case CT_SCALAR:
			SG_ERROR("read_cont_begin_wrapped(): Implementation error"
					 " during writing HDF5File (0)!");
			return false;
		case CT_MATRIX: *len_read_x = m->dims[1]; /* break;  */
		case CT_VECTOR: *len_read_y = m->dims[0]; break;
		}

		return true;
	}

	if (!attr_exists(STR_IS_CONT)) return false;

	string_t ctype_buf, buf;
	type->to_string(ctype_buf, STRING_LEN);
	if (!attr_read_string(STR_CTYPE_NAME, buf, STRING_LEN))
		return false;
	if (strcmp(ctype_buf, buf) != 0) return false;

	switch (type->m_ctype) {
	case CT_SCALAR:
		SG_ERROR("read_cont_begin_wrapped(): Implementation error"
				 " during writing HDF5File (1)!");
		return false;
	case CT_MATRIX:
		if (!attr_read_scalar(TYPE_INDEX, STR_LENGTH_X, len_read_x))
			return false;
		/* break;  */
	case CT_VECTOR:
		if (!attr_read_scalar(TYPE_INDEX, STR_LENGTH_Y, len_read_y))
			return false;
		break;
	}

	return true;
}

bool
CSerializableHDF5File::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	return true;
}

bool
CSerializableHDF5File::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	return true;
}

bool
CSerializableHDF5File::write_string_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	type_item_t* m = m_stack_type.back();

	m->vltype.len = length;

	return true;
}

bool
CSerializableHDF5File::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	type_item_t* m = m_stack_type.back();
	hsize_t buf;

	if (H5Dvlen_get_buf_size(m->dset, m->dtype, m->dspace, &buf) < 0)
		return false;

	m->vltype.len = buf;
	*length = buf/type->sizeof_ptype();

	return true;
}

bool
CSerializableHDF5File::write_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableHDF5File::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableHDF5File::write_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	type_item_t* m = m_stack_type.back();

	if (y > 0) m->vltype.len = 0;

	return true;
}

bool
CSerializableHDF5File::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	type_item_t* m = m_stack_type.back();

	if (y > 0) m->vltype.len = 0;

	return true;
}

bool
CSerializableHDF5File::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableHDF5File::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableHDF5File::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	type_item_t* m_prev = m_stack_type.back();
	type_item_t* m = new type_item_t(m_stack_type.back()->name);
	m_stack_type.push_back(m);

	if (m_prev->y == 0) {
		hbool_t bool_buf = true;
		if (!group_create(m->name, STR_GROUP_PREFIX)) return false;

		if (!attr_write_scalar(H5T_NATIVE_HBOOL, STR_IS_SPARSE,
							   &bool_buf)) return false;
	} else {
		if (!group_open(m->name, STR_GROUP_PREFIX)) return false;
		if (!attr_exists(STR_IS_SPARSE)) return false;
	}

	m->rank = 1; m->dims[0] = length;
	if (m->dims[0] == 0) m->dspace = H5Screate(H5S_NULL);

	if (m->dspace < 0 && (m->dspace = H5Screate_simple(
							  m->rank, m->dims, NULL)) < 0)
		return false;
	if ((m->dtype = new_sparseentrytype(type->m_ptype)) < 0)
		return false;

	string_t name;
	index2string(name, STRING_LEN, type->m_ctype, m_prev->y,
				 m_prev->x);
	if ((m->dset = H5Dcreate2(
			 m_stack_h5stream.back(), name, m->dtype, m->dspace,
			 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0)
		return false;

	/* ************************************************************ */

	char* buf = new char[sizeof_sparsetype()];

	memcpy(buf, &vec_index, sizeof (index_t));

	hid_t mem_type_id;
	if ((mem_type_id = new_sparsetype()) < 0) return false;

	hid_t mem_space_id;
	if ((mem_space_id = H5Screate_simple(0, NULL, NULL)) < 0)
		return false;

	hobj_ref_t* sparse_ref = get_ref_sparstype(buf);
	if (H5Rcreate(sparse_ref, m_stack_h5stream.back(), name,
				  H5R_OBJECT, -1) < 0) return false;

	if (H5Dwrite(m_prev->dset, mem_type_id, mem_space_id,
				 m_prev->dspace, H5P_DEFAULT, buf) < 0)
		return false;

	if (H5Sclose(mem_space_id) < 0) return false;
	if (H5Tclose(mem_type_id) < 0) return false;

	delete buf;

	return true;
}

bool
CSerializableHDF5File::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t* length)
{
	type_item_t* m_prev = m_stack_type.back();
	type_item_t* m = new type_item_t(m_stack_type.back()->name);
	m_stack_type.push_back(m);

	if (!group_open(m->name, STR_GROUP_PREFIX)) return false;
	if (!attr_exists(STR_IS_SPARSE)) return false;

	string_t name;
	index2string(name, STRING_LEN, type->m_ctype, m_prev->y,
				 m_prev->x);
	if ((m->dset = H5Dopen2(m_stack_h5stream.back(), name,
							H5P_DEFAULT)) < 0)
		return false;

	if ((m->dtype = H5Dget_type(m->dset)) < 0) return false;
	if (!isequal_stype2hdf5(type->m_stype, type->m_ptype, m->dtype))
		return false;

	if ((m->dspace = H5Dget_space(m->dset)) < 0) return false;
	if (H5Sget_simple_extent_ndims(m->dspace) != 1) return false;


	if ((m->rank = H5Sget_simple_extent_dims(m->dspace, m->dims, NULL)
			) < 0) return false;

	if (H5Sget_simple_extent_type(m->dspace) != H5S_NULL
		&& m->rank != 1) return false;

	*length = m->dims[0];

	/* ************************************************************ */

	char* buf = new char[sizeof_sparsetype()];

	hid_t mem_type_id;
	if ((mem_type_id = new_sparsetype()) < 0) return false;

	hid_t mem_space_id;
	if ((mem_space_id = H5Screate_simple(0, NULL, NULL)) < 0)
		return false;

	if (H5Dread(m_prev->dset, mem_type_id, mem_space_id,
				m_prev->dspace, H5P_DEFAULT, buf) < 0) return false;

	if (H5Sclose(mem_space_id) < 0) return false;
	if (H5Tclose(mem_type_id) < 0) return false;

	memcpy(vec_index, buf, sizeof (index_t));

	delete buf;

	return true;
}

bool
CSerializableHDF5File::write_sparse_end_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	if (!group_close()) return false;
	delete m_stack_type.back(); m_stack_type.pop_back();

	return true;
}

bool
CSerializableHDF5File::read_sparse_end_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t length)
{
	if (!group_close()) return false;
	delete m_stack_type.back(); m_stack_type.pop_back();

	return true;
}

bool
CSerializableHDF5File::write_sparseentry_begin_wrapped(
	const TSGDataType* type, index_t feat_index, index_t y)
{
	type_item_t* m = m_stack_type.back();
	m->sparse_buf.feat_index = feat_index;

	if (!dspace_select(CT_VECTOR, y, 0)) return false;

	return true;
}

bool
CSerializableHDF5File::read_sparseentry_begin_wrapped(
	const TSGDataType* type, index_t* feat_index, index_t y)
{
	if (!dspace_select(CT_VECTOR, y, 0)) return false;

	return true;
}

bool
CSerializableHDF5File::write_sparseentry_end_wrapped(
	const TSGDataType* type, index_t feat_index, index_t y)
{
	return true;
}

bool
CSerializableHDF5File::read_sparseentry_end_wrapped(
	const TSGDataType* type, index_t* feat_index, index_t y)
{
	type_item_t* m = m_stack_type.back();
	*feat_index = m->sparse_buf.feat_index;

	return true;
}

bool
CSerializableHDF5File::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	type_item_t* m = m_stack_type.back();
	m->y = y; m->x = x;

	if (type->m_ptype != PT_SGSERIALIZABLE_PTR) {
		if (!dspace_select(type->m_ctype, y, x)) return false;
		return true;
	}

	string_t name;
	if (!index2string(name, STRING_LEN, type->m_ctype, y, x))
		return false;
	if (!group_create(name, "")) return false;

	return true;
}

bool
CSerializableHDF5File::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	type_item_t* m = m_stack_type.back();
	m->y = y; m->x = x;

	if (type->m_ptype != PT_SGSERIALIZABLE_PTR) {
		if (!dspace_select(type->m_ctype, y, x)) return false;
		return true;
	}

	string_t name;
	if (!index2string(name, STRING_LEN, type->m_ctype, y, x))
		return false;
	if (!group_open(name, "")) return false;

	return true;
}

bool
CSerializableHDF5File::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (type->m_ptype == PT_SGSERIALIZABLE_PTR)
		if (!group_close()) return false;

	return true;
}

bool
CSerializableHDF5File::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (type->m_ptype == PT_SGSERIALIZABLE_PTR)
		if (!group_close()) return false;

	return true;
}

bool
CSerializableHDF5File::write_sgserializable_begin_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	hbool_t bool_buf = true;

	if (!attr_write_scalar(H5T_NATIVE_HBOOL, STR_IS_SGSERIALIZABLE,
						   &bool_buf)) return false;

	if (*sgserializable_name == '\0') {
		if (!attr_write_scalar(H5T_NATIVE_HBOOL, STR_IS_NULL,
							   &bool_buf))
			return false;
		return true;
	}

	if (!attr_write_string(STR_INSTANCE_NAME, sgserializable_name))
		return false;

	if (generic != PT_NOT_GENERIC) {
		string_t buf;
		TSGDataType::ptype_to_string(buf, generic, STRING_LEN);
		if (!attr_write_string(STR_GENERIC_NAME, buf)) return false;
	}

	return true;
}

bool
CSerializableHDF5File::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	if (!attr_exists(STR_IS_SGSERIALIZABLE)) return false;

	if (attr_exists(STR_IS_NULL)) {
		*sgserializable_name = '\0'; return true;
	}

	if (!attr_read_string(STR_INSTANCE_NAME, sgserializable_name,
						  STRING_LEN)) return false;

	if (attr_exists(STR_GENERIC_NAME)) {
		string_t buf;
		if (!attr_read_string(STR_GENERIC_NAME, buf, STRING_LEN))
			return false;
		if (!TSGDataType::string_to_ptype(generic, buf))
			return false;
	}

	return true;
}

bool
CSerializableHDF5File::write_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableHDF5File::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableHDF5File::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	type_item_t* m = new type_item_t(name); m_stack_type.push_back(m);

	if (type->m_ptype == PT_SGSERIALIZABLE_PTR) {
		if (!group_create(name, "")) return false;
		return true;
	}

	switch (type->m_ctype) {
	case CT_SCALAR:
		m->rank = 0; break;
	case CT_VECTOR:
		m->rank = 1; m->dims[0] = *type->m_length_y;
		if (m->dims[0] == 0) m->dspace = H5Screate(H5S_NULL);
		break;
	case CT_MATRIX:
		m->rank = 2;
		m->dims[0] = *type->m_length_y; m->dims[1] = *type->m_length_x;
		if (m->dims[0] *m->dims[1] == 0)
			m->dspace = H5Screate(H5S_NULL);
		break;
	}

	if (m->dspace < 0 && (m->dspace = H5Screate_simple(
							  m->rank, m->dims, NULL)) < 0)
		return false;
	if ((m->dtype = new_stype2hdf5(type->m_stype, type->m_ptype)) < 0)
		return false;

	if ((m->dset = H5Dcreate2(
			 m_stack_h5stream.back(), name, m->dtype, m->dspace,
			 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0)
		return false;

	return true;
}

bool
CSerializableHDF5File::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	type_item_t* m = new type_item_t(name); m_stack_type.push_back(m);

	if (type->m_ptype == PT_SGSERIALIZABLE_PTR) {
		if (!group_open(name, "")) return false;
		return true;
	}

	if ((m->dset = H5Dopen2(m_stack_h5stream.back(), name,
							H5P_DEFAULT)) < 0)
		return false;

	if ((m->dtype = H5Dget_type(m->dset)) < 0) return false;
	if (!isequal_stype2hdf5(type->m_stype, type->m_ptype, m->dtype))
		return false;

	if ((m->dspace = H5Dget_space(m->dset)) < 0) return false;

	if (H5Sget_simple_extent_ndims(m->dspace) > 2) return false;
	if ((m->rank = H5Sget_simple_extent_dims(m->dspace, m->dims, NULL)
			) < 0) return false;
	switch (type->m_ctype) {
	case CT_SCALAR: if (m->rank != 0) return false; break;
	case CT_VECTOR:
		if (H5Sget_simple_extent_type(m->dspace) != H5S_NULL
			&& m->rank != 1) return false;
		break;
	case CT_MATRIX:
		if (H5Sget_simple_extent_type(m->dspace) != H5S_NULL
			&& m->rank != 2) return false;
		break;
	}

	return true;
}

bool
CSerializableHDF5File::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (type->m_ptype == PT_SGSERIALIZABLE_PTR)
		if (!group_close()) return false;

	delete m_stack_type.back(); m_stack_type.pop_back();
	return true;
}

bool
CSerializableHDF5File::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (type->m_ptype == PT_SGSERIALIZABLE_PTR)
		if (!group_close()) return false;

	delete m_stack_type.back(); m_stack_type.pop_back();
	return true;
}

#endif /* HAVE_HDF5  */
