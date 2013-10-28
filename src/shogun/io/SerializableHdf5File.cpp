/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/lib/config.h>
#ifdef HAVE_HDF5

#include <shogun/io/SerializableHdf5File.h>
#include <shogun/io/SerializableHdf5Reader00.h>

#define NOT_OPEN                   ((hid_t) -1)

#define STR_KEY_FILETYPE           "filetype"
#define STR_FILETYPE_00 \
	"_SHOGUN_SERIALIZABLE_HDF5_FILE_V_00_"

using namespace shogun;

CSerializableHdf5File::type_item_t::type_item_t(const char* name_)
{
	rank = 0;
	dims[0] = dims[1] = 0;
	dspace = dtype = dset = NOT_OPEN;
	vltype = NULL;
	y = x = sub_y = 0;
	sparse_ptr = NULL;
	name = name_;
}

CSerializableHdf5File::type_item_t::~type_item_t()
{
	if (dset >= 0) H5Dclose(dset);
	if (dtype >= 0) H5Tclose(dtype);
	if (dspace >= 0) H5Sclose(dspace);
	if (vltype != NULL) SG_FREE(vltype);
	/* Do not delete SPARSE_PTR  */
}

hid_t
CSerializableHdf5File::sizeof_sparsetype() {
	return H5Tget_size(TYPE_INDEX) + H5Tget_size(H5T_STD_REF_OBJ);
}
hid_t
CSerializableHdf5File::new_sparsetype()
{
	hid_t result = H5Tcreate(H5T_COMPOUND, sizeof_sparsetype());

	if (H5Tinsert(result, STR_SPARSE_FPTR, H5Tget_size(TYPE_INDEX),
				  H5T_STD_REF_OBJ) < 0)
		return NOT_OPEN;

	return result;
}
hobj_ref_t*
CSerializableHdf5File::get_ref_sparstype(void* sparse_buf) {
	return (hobj_ref_t*)
		((char*) sparse_buf + H5Tget_size(TYPE_INDEX));
}

hid_t
CSerializableHdf5File::new_sparseentrytype(EPrimitiveType ptype)
{
	hid_t result = H5Tcreate(H5T_COMPOUND,
							 TSGDataType::sizeof_sparseentry(ptype));
	if (result < 0) return NOT_OPEN;

	if (H5Tinsert(result, STR_SPARSEENTRY_FINDEX,
				  HOFFSET(SGSparseVectorEntry<char>, feat_index), TYPE_INDEX)
		< 0) return NOT_OPEN;
	if (H5Tinsert(result, STR_SPARSEENTRY_ENTRY, TSGDataType
				  ::offset_sparseentry(ptype),
				  ptype2hdf5(ptype)) < 0) return NOT_OPEN;

	return result;
}

hid_t
CSerializableHdf5File::ptype2hdf5(EPrimitiveType ptype)
{
	switch (ptype) {
	case PT_BOOL:
		switch (sizeof (bool)) {
		case 1: return H5T_NATIVE_UINT8;
		case 2: return H5T_NATIVE_UINT16;
		case 4: return H5T_NATIVE_UINT32;
		case 8: return H5T_NATIVE_UINT64;
		default: break;
		}
		break;
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
	case PT_COMPLEX128: return NOT_OPEN; break;
	case PT_SGOBJECT: return NOT_OPEN; break;
	case PT_UNDEFINED:
		SG_SERROR("Type undefined\n");
		return NOT_OPEN;
	}

	return NOT_OPEN;
}

hid_t
CSerializableHdf5File::new_stype2hdf5(EStructType stype,
									  EPrimitiveType ptype)
{
	hid_t result = ptype2hdf5(ptype);

	switch (stype) {
	case ST_NONE: result = H5Tcopy(result); break;
	case ST_STRING: result = H5Tvlen_create(result); break;
	case ST_SPARSE: result = new_sparsetype(); break;
	default: break;
	}

	return result;
}

bool
CSerializableHdf5File::index2string(
	char* dest, size_t n, EContainerType ctype, index_t y, index_t x)
{
	switch (ctype) {
	case CT_NDARRAY: SG_SNOTIMPLEMENTED
	case CT_SCALAR: return false;
	case CT_VECTOR: case CT_SGVECTOR: snprintf(dest, n, "y%u", y); break;
	case CT_MATRIX: case CT_SGMATRIX: snprintf(dest, n, "y%u_x%u", y, x); break;
	default: return false;
	}

	return true;
}

bool
CSerializableHdf5File::isequal_stype2hdf5(EStructType stype,
										  EPrimitiveType ptype,
										  hid_t htype)
{
	hid_t pbuf = ptype2hdf5(ptype), pbuf2 = NOT_OPEN;

	bool to_close = false;
	switch (stype) {
	case ST_UNDEFINED:
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
CSerializableHdf5File::dspace_select(EContainerType ctype, index_t y,
									 index_t x)
{
	type_item_t* m = m_stack_type.back();

	if (H5Sselect_none(m->dspace) < 0) return false;

	hsize_t coord[2];
	switch (ctype) {
	case CT_NDARRAY: SG_NOTIMPLEMENTED
	case CT_SCALAR: return false;
	case CT_MATRIX: case CT_SGMATRIX: coord[1] = x; /* break;  */
	case CT_VECTOR: case CT_SGVECTOR: coord[0] = y; break;
	default: return false;
	}
	if (H5Sselect_elements(m->dspace, H5S_SELECT_SET, 1, coord) < 0)
		return false;

	return true;
}

bool
CSerializableHdf5File::attr_write_scalar(
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
CSerializableHdf5File::attr_write_string(
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
CSerializableHdf5File::attr_exists(const char* name)
{
	return H5Aexists(m_stack_h5stream.back(), name) > 0;
}

size_t
CSerializableHdf5File::attr_get_size(const char* name)
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
CSerializableHdf5File::attr_read_scalar(
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
CSerializableHdf5File::attr_read_string(
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
CSerializableHdf5File::group_create(const char* name,
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
CSerializableHdf5File::group_open(const char* name,
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
CSerializableHdf5File::group_close()
{
	if (H5Gclose(m_stack_h5stream.back()) < 0) return false;
	m_stack_h5stream.pop_back();

	return true;
}

CSerializableHdf5File::CSerializableHdf5File()
	:CSerializableFile() { init(""); }

CSerializableHdf5File::CSerializableHdf5File(const char* fname, char rw)
	:CSerializableFile()
{
	CSerializableFile::init(NULL, rw, fname);
	init(fname);
}

CSerializableHdf5File::~CSerializableHdf5File()
{
	while (m_stack_type.get_num_elements() > 0) {
		delete m_stack_type.back(); m_stack_type.pop_back();
	}

	close();
}

CSerializableFile::TSerializableReader*
CSerializableHdf5File::new_reader(char* dest_version, size_t n)
{
	if (!attr_read_string(STR_KEY_FILETYPE, dest_version, n))
		return NULL;

	if (strcmp(STR_FILETYPE_00, dest_version) == 0)
		return new SerializableHdf5Reader00(this);

	return NULL;
}

void
CSerializableHdf5File::init(const char* fname)
{
	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n")
		close(); return;
	}

	hid_t h5stream = NOT_OPEN;
	switch (m_task) {
	case 'w':
		h5stream = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT,
							 H5P_DEFAULT);
		break;
	case 'r':
		h5stream = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
		break;
	default:
		SG_WARNING("Could not open file `%s', unknown mode!\n",
				   m_filename);
		close(); return;
	}

	if (h5stream < 0) {
		SG_WARNING("Could not open file `%s'!\n", m_filename)
		close(); return;
	}

	m_stack_h5stream.push_back(h5stream);
	switch (m_task) {
	case 'w':
		if (!attr_write_string(STR_KEY_FILETYPE, STR_FILETYPE_00)) {
			SG_WARNING("%s: Could not open file for writing during "
					   "writing filetype!\n", fname);
			close(); return;
		}
		break;
	case 'r': break;
	default: break;
	}
}

void
CSerializableHdf5File::close()
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
CSerializableHdf5File::is_opened()
{
	return m_stack_h5stream.get_num_elements() > 0;
}

bool
CSerializableHdf5File::write_scalar_wrapped(
	const TSGDataType* type, const void* param)
{
	type_item_t* m = m_stack_type.back();

	switch (type->m_stype) {
	case ST_NONE:
		if (m->y != 0 || m->x != 0) return true;
		break;
	case ST_STRING:
		if (m->sub_y == 0)
			m->vltype[m->x*m->dims[1] + m->y].p = (void*) param;

		if ((m->sub_y
			 < (index_t) m->vltype[m->x*m->dims[1] + m->y].len-1)
			|| ((type->m_ctype == CT_VECTOR || type->m_ctype == CT_SGVECTOR) && m->y
				< (index_t) m->dims[0]-1)
			|| ((type->m_ctype == CT_MATRIX || type->m_ctype==CT_SGMATRIX)
				&& (m->x < (index_t) m->dims[0]-1
					|| m->y < (index_t) m->dims[1]-1)))
			return true;
		break;
	case ST_SPARSE:
		if (m->sub_y != 0) return true;
		break;
	default: return false;
	}

	hid_t mem_type_id;
	if ((mem_type_id = new_stype2hdf5(type->m_stype, type->m_ptype)
			) < 0) return false;

	switch (type->m_stype) {
	case ST_NONE:
		if (H5Dwrite(m->dset, mem_type_id, H5S_ALL, H5S_ALL,
					 H5P_DEFAULT, param) < 0) return false;
		break;
	case ST_STRING:
		if (H5Dwrite(m->dset, mem_type_id, H5S_ALL, H5S_ALL,
					 H5P_DEFAULT, m->vltype) < 0) return false;
		break;
	case ST_SPARSE:
		if (H5Dwrite(m->dset, m->dtype, H5S_ALL, H5S_ALL,
					 H5P_DEFAULT, m->sparse_ptr) < 0) return false;
		break;
	default: return false;
	}

	if (H5Tclose(mem_type_id) < 0) return false;

	return true;
}

bool
CSerializableHdf5File::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	hbool_t bool_buf = true;

	if (type->m_ptype != PT_SGOBJECT) return true;

	if (!attr_write_scalar(H5T_NATIVE_HBOOL, STR_IS_CONT, &bool_buf))
		return false;

	string_t ctype_buf;
	type->to_string(ctype_buf, STRING_LEN);
	if (!attr_write_string(STR_CTYPE_NAME, ctype_buf)) return false;

	switch (type->m_ctype) {
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
	case CT_SCALAR:
		SG_ERROR("write_cont_begin_wrapped(): Implementation error "
				 "during writing Hdf5File!");
		return false;
	case CT_MATRIX: case CT_SGMATRIX:
		if (!attr_write_scalar(TYPE_INDEX, STR_LENGTH_X, &len_real_x))
			return false;
		/* break;  */
	case CT_VECTOR: case CT_SGVECTOR:
		if (!attr_write_scalar(TYPE_INDEX, STR_LENGTH_Y, &len_real_y))
			return false;
		break;
	default: return false;
	}

	return true;
}

bool
CSerializableHdf5File::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	return true;
}

bool
CSerializableHdf5File::write_string_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	type_item_t* m = m_stack_type.back();

	m->vltype[m->x*m->dims[1] + m->y].len = length;

	return true;
}

bool
CSerializableHdf5File::write_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableHdf5File::write_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	type_item_t* m = m_stack_type.back();

	m->sub_y = y;

	return true;
}

bool
CSerializableHdf5File::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableHdf5File::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	type_item_t* m_prev = m_stack_type.back();

	if(!dspace_select(type->m_ctype, m_prev->y, m_prev->x))
		return false;

	type_item_t* m = new type_item_t(m_stack_type.back()->name);
	m_stack_type.push_back(m);

	/* ************************************************************ */

	if (m_prev->y == 0 && m_prev->x == 0) {
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

	char* buf = SG_MALLOC(char, sizeof_sparsetype());

	hid_t mem_type_id;
	if ((mem_type_id = new_sparsetype()) < 0) return false;

	hid_t mem_space_id;
	if ((mem_space_id = H5Screate_simple(0, NULL, NULL)) < 0)
		return false;

	hobj_ref_t* sparse_ref = get_ref_sparstype(buf);
	if (H5Rcreate(sparse_ref, m_stack_h5stream.back(), name,
				  H5R_OBJECT, -1) < 0) return false;

	if (H5Dwrite(m_prev->dset, mem_type_id, mem_space_id,
				 m_prev->dspace, H5P_DEFAULT, buf) < 0) return false;

	if (H5Sclose(mem_space_id) < 0) return false;
	if (H5Tclose(mem_type_id) < 0) return false;

	delete buf;

	return true;
}

bool
CSerializableHdf5File::write_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (!group_close()) return false;
	delete m_stack_type.back(); m_stack_type.pop_back();

	return true;
}

bool
CSerializableHdf5File::write_sparseentry_begin_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	type_item_t* m = m_stack_type.back();

	m->sparse_ptr = (SGSparseVectorEntry<char>*) first_entry;
	m->sub_y = y;

	return true;
}

bool
CSerializableHdf5File::write_sparseentry_end_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	return true;
}

bool
CSerializableHdf5File::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	type_item_t* m = m_stack_type.back();
	m->y = y; m->x = x;

	if (type->m_ptype != PT_SGOBJECT) return true;

	string_t name;
	if (!index2string(name, STRING_LEN, type->m_ctype, y, x))
		return false;
	if (!group_create(name, "")) return false;

	return true;
}

bool
CSerializableHdf5File::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (type->m_ptype == PT_SGOBJECT)
		if (!group_close()) return false;

	return true;
}

bool
CSerializableHdf5File::write_sgserializable_begin_wrapped(
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
CSerializableHdf5File::write_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableHdf5File::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	type_item_t* m = new type_item_t(name); m_stack_type.push_back(m);

	if (type->m_ptype == PT_SGOBJECT) {
		if (!group_create(name, "")) return false;
		return true;
	}

	switch (type->m_ctype) {
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
	case CT_SCALAR:
		m->rank = 0;
		if (type->m_stype == ST_STRING) m->vltype = SG_MALLOC(hvl_t, 1);
		break;
	case CT_VECTOR: case CT_SGVECTOR:
		m->rank = 1; m->dims[0] = *type->m_length_y;
		if (m->dims[0] == 0) m->dspace = H5Screate(H5S_NULL);
		if (type->m_stype == ST_STRING)
			m->vltype = SG_MALLOC(hvl_t, m->dims[0]);
		break;
	case CT_MATRIX: case CT_SGMATRIX:
		m->rank = 2;
		m->dims[0] = *type->m_length_x; m->dims[1] = *type->m_length_y;
		if (m->dims[0] *m->dims[1] == 0)
			m->dspace = H5Screate(H5S_NULL);
		if (type->m_stype == ST_STRING)
			m->vltype = SG_MALLOC(hvl_t, m->dims[0] *m->dims[1]);
		break;
	default: return false;
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
CSerializableHdf5File::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (type->m_ptype == PT_SGOBJECT)
		if (!group_close()) return false;

	delete m_stack_type.back(); m_stack_type.pop_back();
	return true;
}

#endif /* HAVE_HDF5  */
