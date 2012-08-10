/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

/* helper's stuff */

%define BUFFER_VECTOR_INFO(type_name)
%wrapper
%{

struct buffer_vector_ ## type_name ## _info
{
	SGVector< type_name >  buf; 
	Py_ssize_t* shape;
	Py_ssize_t* strides;
};

%}
%enddef // BUFFER_VECTOR_INFO

%define BUFFER_MATRIX_INFO(type_name)
%wrapper
%{

struct buffer_matrix_ ## type_name ## _info
{
	SGMatrix< type_name >  buf; 
	Py_ssize_t* shape;
	Py_ssize_t* strides;
};

%}
%enddef // BUFFER_MATRIX_INFO

BUFFER_VECTOR_INFO(bool)
BUFFER_VECTOR_INFO(char)
BUFFER_VECTOR_INFO(uint8_t)
BUFFER_VECTOR_INFO(uint16_t)
BUFFER_VECTOR_INFO(int16_t)
BUFFER_VECTOR_INFO(int32_t)
BUFFER_VECTOR_INFO(uint32_t)
BUFFER_VECTOR_INFO(int64_t)
BUFFER_VECTOR_INFO(uint64_t)
BUFFER_VECTOR_INFO(float32_t)
BUFFER_VECTOR_INFO(float64_t)

BUFFER_MATRIX_INFO(bool)
BUFFER_MATRIX_INFO(char)
BUFFER_MATRIX_INFO(uint8_t)
BUFFER_MATRIX_INFO(uint16_t)
BUFFER_MATRIX_INFO(int16_t)
BUFFER_MATRIX_INFO(int32_t)
BUFFER_MATRIX_INFO(uint32_t)
BUFFER_MATRIX_INFO(int64_t)
BUFFER_MATRIX_INFO(uint64_t)
BUFFER_MATRIX_INFO(float32_t)
BUFFER_MATRIX_INFO(float64_t)

%wrapper
%{

struct buffer_info
{
	Py_ssize_t* shape;
	Py_ssize_t* strides;
};

void get_slice_in_bounds(Py_ssize_t* ilow, Py_ssize_t* ihigh, Py_ssize_t max_idx)
{
	if (*ilow<0)
	{
		*ilow=0;
	}
	else if (*ilow>max_idx)
	{
		*ilow = max_idx;
	}
	if (*ihigh<*ilow)
	{
		*ihigh=*ilow;
	}
	else if (*ihigh>max_idx)
	{
		*ihigh=max_idx;
	}
}

Py_ssize_t get_idx_in_bounds(Py_ssize_t idx, Py_ssize_t max_idx)
{
	if (idx>=max_idx || idx<-max_idx)
	{
		PyErr_SetString(PyExc_IndexError, "index out of bounds");
		return -1;
	}
	else if (idx<0)
		return idx+max_idx;

	return idx;
}

int parse_tuple_item(PyObject* item, Py_ssize_t length,
						Py_ssize_t* ilow, Py_ssize_t* ihigh,
						Py_ssize_t* step, Py_ssize_t* slicelength)
{
	if (PySlice_Check(item))
	{
		PySlice_GetIndicesEx((PySliceObject*) item, length, ilow, ihigh, step, slicelength);
		get_slice_in_bounds(ilow, ihigh, length);

		return 2;
	}
	else if (PyInt_Check(item) || PyArray_IsScalar(item, Integer) ||
       		PyLong_Check(item) || (PyIndex_Check(item) && !PySequence_Check(item)))
	{
		npy_intp idx;
		idx = PyArray_PyIntAsIntp(item);
		idx = get_idx_in_bounds(idx, length);

		*ilow=idx;
		*ihigh=idx+1;

		return 1;
	}

	return 0;
}

%}
