/* Helper functions */
%wrapper
%{
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
