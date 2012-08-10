/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

#ifdef SWIGPYTHON

%include "protocols_helper.i"

/* Numeric operators for DenseFeatures */
%define NUMERIC_DENSEFEATURES(class_name, type_name, format_str, operator_name, operator)

PyObject* class_name ## _inplace ## operator_name ## (PyObject *self, PyObject *o2)
{
	CDenseFeatures< type_name >* arg1=0; // self in c++ repr

	void* argp1=0; // pointer to self
	int res1=0; // result for self's casting
	int res2=0; // result for checking buffer
	int res3=0; // result for getting buffer

	PyObject* resultobj=0;
	Py_buffer view;
	SGMatrix< type_name > buf; // internal buffer of self

	int num_feat, num_vec; // shape of buffer of self
	Py_ssize_t shape[2];
	Py_ssize_t strides[2];

	type_name* lhs;
	char* rhs;

	res1=SWIG_ConvertPtr(self, &argp1, SWIG_TypeQuery("shogun::CDenseFeatures<type_name>"), 0 |  0 );
	arg1=reinterpret_cast< CDenseFeatures< type_name > * >(argp1);

	res2=PyObject_CheckBuffer(o2);
	if (!res2)
	{
		SWIG_exception_fail(SWIG_ArgError(res2), "this object don't support buffer protocol");
	}

	res3=PyObject_GetBuffer(o2, &view, PyBUF_F_CONTIGUOUS | PyBUF_ND | PyBUF_STRIDES | 0);
	if (res3!=0 || view.buf==NULL)
	{
		SWIG_exception_fail(SWIG_ArgError(res3), "bad buffer");
	}

	// checking that buffer is right
	if (view.ndim!=2)
	{
		SWIG_exception_fail(SWIG_ArgError(view.ndim), "wrong dimension");
	}

	if (view.itemsize!=sizeof(type_name))
	{
		SWIG_exception_fail(SWIG_ArgError(view.itemsize), "wrong type");
	}

	if (view.shape==NULL)
	{
		SWIG_exception_fail(SWIG_ArgError(0), "wrong shape");
	}

	shape[0]=view.shape[0];
	shape[1]=view.shape[1];
	if (shape[0]!=arg1->get_num_features() || shape[1]!=arg1->get_num_vectors())
		SWIG_exception_fail(SWIG_ArgError(0), "wrong size");

	strides[0]=view.strides[0];
	strides[1]=view.strides[1];

	if (view.len!=(shape[0]*shape[1])*view.itemsize)
		SWIG_exception_fail(SWIG_ArgError(view.len), "bad buffer");

	// calculation
	buf=arg1->get_feature_matrix();
	num_feat=arg1->get_num_features();
	num_vec=arg1->get_num_vectors();

	lhs=buf.matrix;
	rhs=(char*) view.buf;

	for (int i=0; i<num_vec; i++)
	{
		for (int j=0; j<num_feat; j++)
		{
			lhs[num_feat*i + j] ## operator ## = (*(type_name*) (rhs + strides[0]*j + strides[1]*i));
		}
	}

	resultobj=self;
	PyBuffer_Release(&view);

	Py_INCREF(resultobj);
	return resultobj;

fail:
	return NULL;
}

%enddef // NUMERIC_DENSEFEATURES

/* Python protocols for DenseFeatures */
%define PYPROTO_DENSEFEATURES(class_name, type_name, format_str, typecode)

%wrapper
%{

/* used by PyObject_GetBuffer */
static int class_name ## _getbuffer(PyObject *self, Py_buffer *view, int flags)
{
	CDenseFeatures< type_name >* arg1=(CDenseFeatures< type_name > *) 0; // self in c++ repr
	void* argp1=0; // pointer to self
	int res1=0; // result for self's casting

	int num_feat=0, num_vec=0;
	Py_ssize_t* shape=NULL;
	Py_ssize_t* strides=NULL;
	
	buffer_matrix_ ## type_name ## _info* info=NULL;

	static char* format=(char *) format_str; // http://docs.python.org/dev/library/struct.html#module-struct

	res1 = SWIG_ConvertPtr(self, &argp1, SWIG_TypeQuery("shogun::CDenseFeatures<type_name>"), 0 |  0 );
	if (!SWIG_IsOK(res1))
	{
		SWIG_exception_fail(SWIG_ArgError(res1),
					"in method '" " class_name _getbuffer" "', argument " "1"" of type '" "CDenseFeatures< type_name > *""'");
	}

	if ((flags & PyBUF_C_CONTIGUOUS)==PyBUF_C_CONTIGUOUS)
	{
		PyErr_SetString(PyExc_ValueError, "class_name is not C-contiguous");
		goto fail;
	}

	if ((flags & PyBUF_STRIDES)!=PyBUF_STRIDES &&
		(flags & PyBUF_ND)==PyBUF_ND)
	{
		PyErr_SetString(PyExc_ValueError, "class_name is not C-contiguous");
		goto fail;
	}

	arg1=reinterpret_cast< CDenseFeatures < type_name >* >(argp1);

	info=new buffer_matrix_ ## type_name ## _info;

	info->buf=arg1->get_feature_matrix();
	num_feat=arg1->get_num_features();
	num_vec=arg1->get_num_vectors();

	view->buf=info->buf.matrix;

	shape=new Py_ssize_t[2];
	shape[0]=num_feat;
	shape[1]=num_vec;

	strides=new Py_ssize_t[2];
	strides[0]=sizeof( type_name );
	strides[1]=sizeof( type_name ) * num_feat;

	info->shape=shape;
	info->strides=strides;

	view->ndim=2;

	view->format=(char*) format_str;
	view->itemsize=sizeof( type_name );

	view->len=(shape[0]*shape[1])*view->itemsize;
	view->shape=shape;
	view->strides=strides;

	view->readonly=0;
	view->suboffsets=NULL;
	view->internal=(void*) info;

	view->obj=(PyObject*) self;
	Py_INCREF(self);

	return 0;

fail:
	view->obj=NULL;
	return -1;
}

/* used by PyBuffer_Release */
static void class_name ## _releasebuffer(PyObject *self, Py_buffer *view)
{
	buffer_matrix_ ## type_name ## _info* temp=NULL;
	if (view->obj!=NULL && view->internal!=NULL)
	{
		temp=(buffer_matrix_ ## type_name ## _info*) view->internal;
		if (temp->shape!=NULL)
			delete[] temp->shape;

		if (temp->strides!=NULL)
			delete[] temp->strides;

		temp->buf=SGMatrix< type_name >();
		delete temp;
	}
}

/* used by PySequence_GetItem */
static PyObject* class_name ## _getitem(PyObject *self, Py_ssize_t idx)
{
	CDenseFeatures< type_name >* arg1=0; // self in c++ repr
	void* argp1=0; // pointer to self
	int res1=0; // result for self's casting

	char* data=0; // internal data of self
	int num_feat=0, num_vec=0;

	SGMatrix< type_name > temp;

	Py_ssize_t* shape;
	Py_ssize_t* strides;

	PyArrayObject* ret;
	PyArray_Descr* descr=PyArray_DescrFromType(typecode);

	res1 = SWIG_ConvertPtr(self, &argp1, SWIG_TypeQuery("shogun::CDenseFeatures<type_name>"), 0 |  0 );
	if (!SWIG_IsOK(res1))
	{
		SWIG_exception_fail(SWIG_ArgError(res1),
					"in method '" " class_name _getitem" "', argument " "1"" of type '" "CDenseFeatures< type_name > *""'");
	}

	arg1=reinterpret_cast< CDenseFeatures< type_name >* >(argp1);
	
	temp=arg1->get_feature_matrix();
	num_feat=arg1->get_num_features();
	num_vec=arg1->get_num_vectors();

	data=(char*) temp.matrix;

	idx=get_idx_in_bounds(idx, num_feat);
	if (idx < 0)
	{
		goto fail;
	}

	data+=idx * sizeof( type_name );

	shape=new Py_ssize_t[2];
	shape[0]=1;
	shape[1]=num_vec;

	strides=new Py_ssize_t[2];
	strides[0]=sizeof( type_name );
	strides[1]=sizeof( type_name ) * num_feat;

	ret=(PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, descr,
					1, shape+1,
					strides+1, data,
 					NPY_FARRAY | NPY_WRITEABLE,
 					(PyObject *) self);
	if (ret==NULL)
		goto fail;

	Py_INCREF(self);
	return (PyObject*)ret;

fail:
	return NULL;
}

/* used by PySequence_SetItem */
static int class_name ## _setitem(PyObject *self, Py_ssize_t idx, PyObject *v)
{
	PyArrayObject* tmp=NULL;
	int ret=0;

	if (v==NULL)
	{
		// TODO error message
		goto fail;
	}

	tmp=(PyArrayObject *) class_name ## _getitem(self, idx);
	if(tmp==NULL)
	{
		goto fail;
	}
	ret=PyArray_CopyObject(tmp, v);
	Py_DECREF(tmp);
	return ret;

fail:
	return -1;
}


/* used by PySequence_GetSlice */
static PyObject* class_name ## _getslice(PyObject *self, Py_ssize_t ilow, Py_ssize_t ihigh)
{
	CDenseFeatures< type_name >* arg1=0; // self in c++ repr
	void* argp1=0; // pointer to self
	int res1=0 ; // result for self's casting

	int num_feat=0, num_vec=0;
	char* data=0; // internal data of self

	SGMatrix< type_name > temp;

	Py_ssize_t* shape;
	Py_ssize_t* strides;

	PyArrayObject* ret;
	PyArray_Descr* descr=PyArray_DescrFromType(typecode);

	res1=SWIG_ConvertPtr(self, &argp1, SWIG_TypeQuery("shogun::CDenseFeatures<type_name>"), 0 |  0 );
	if (!SWIG_IsOK(res1))
	{
		SWIG_exception_fail(SWIG_ArgError(res1),
					"in method '" " class_name _slice" "', argument " "1"" of type '" "CDenseFeatures< type_name > *""'");
	}

	arg1=reinterpret_cast< CDenseFeatures< type_name >* >(argp1);

	temp=arg1->get_feature_matrix();
	num_feat=arg1->get_num_features();
	num_vec=arg1->get_num_vectors();

	data=(char*) temp.matrix;

	get_slice_in_bounds(&ilow, &ihigh, num_feat);
	if (ilow < ihigh)
	{
		data+=ilow * sizeof( type_name );
	}

	shape=new Py_ssize_t[2];
	shape[0]=ihigh - ilow;
	shape[1]=num_vec;

	strides=new Py_ssize_t[2];
	strides[0]=sizeof( type_name );
	strides[1]=sizeof( type_name ) * num_feat;

	ret=(PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, descr,
					2, shape,
 					strides, data,
 					NPY_FARRAY | NPY_WRITEABLE,
 					(PyObject *) self);
	if (ret==NULL)
		goto fail;

	Py_INCREF(self);
	return (PyObject *) ret;

fail:
	return NULL;
}

/* used by PySequence_SetSlice */
static int class_name ## _setslice(PyObject *self, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject* v)
{
	PyArrayObject* tmp=NULL;
	int ret=0;

	if (v==NULL)
	{
		// TODO error message
		goto fail;
	}

	tmp=(PyArrayObject *) class_name ## _getslice(self, ilow, ihigh);
	if(tmp==NULL)
	{
		goto fail;
	}
	ret = PyArray_CopyObject(tmp, v);
	Py_DECREF(tmp);
	return ret;

fail:
	return -1;
}

/* used for numpy's style slicing */
static PyObject* class_name ## _getsubscript(PyObject *self, PyObject *key, bool get_scalar=true)
{
	// key is tuple, like (PySlice or PyLong, PySlice or PyLong)
	// or only PySlice or PyLong

	CDenseFeatures< type_name >* arg1=0; // self in c++ repr
	void* argp1=0; // pointer to self
	int res1=0 ; // result for self's casting

	int num_feat=0;
	int num_vec=0;
	int ndims=2;
	char* data = 0; // internal data of self

	Py_ssize_t* shape;
	Py_ssize_t* strides;

	SGMatrix< type_name > temp;

	PyObject* ret;
	PyArray_Descr* descr=PyArray_DescrFromType(typecode);

	int num_items=0; // size of tuple
	int type_item1=0; // results for tuple parsing
	int type_item2=0;

	Py_ssize_t feat_high=0;
	Py_ssize_t feat_low=0;
	Py_ssize_t vec_high=0;
	Py_ssize_t vec_low=0;

	Py_ssize_t feat_step=0;
	Py_ssize_t vec_step=0;
	Py_ssize_t feat_slicelength=0;
	Py_ssize_t vec_slicelength=0;

	PyObject* tmp; // temporary object for tuple's item

	res1 = SWIG_ConvertPtr(self, &argp1, SWIG_TypeQuery("shogun::CDenseFeatures<type_name>"), 0 |  0 );
	if (!SWIG_IsOK(res1))
	{
		SWIG_exception_fail(SWIG_ArgError(res1),
					"in method '" " class_name _subscript" "', argument " "1"" of type '" "CDenseFeatures< type_name > *""'");
	}

	arg1=reinterpret_cast< CDenseFeatures< type_name >* >(argp1);

	temp=arg1->get_feature_matrix();
	num_feat=arg1->get_num_features();
	num_vec=arg1->get_num_vectors();

	data=(char*) temp.matrix;

	feat_high=num_feat;
	vec_high=num_vec;

	if(PyTuple_Check(key))
	{
		num_items=PyTuple_GET_SIZE(key);
		if (num_items==2)
		{
			// get slice for feat's dim
			tmp=PyTuple_GET_ITEM(key, 0); // first element of tuple
			type_item1=parse_tuple_item(tmp, num_feat,
								&feat_low, &feat_high,
								&feat_step, &feat_slicelength);
			if (type_item1==0)
			{
				goto fail;
			}

			// get slice for vector's dim
			tmp=PyTuple_GET_ITEM(key, 1); // second element of tuple
			type_item2=parse_tuple_item(tmp, num_vec,
								&vec_low, &vec_high,
								&vec_step, &vec_slicelength);
			if (type_item2==0)
			{
				goto fail;
			}
		}
		else
		{
			SWIG_exception_fail(SWIG_ArgError(res1), "same size is needed...");
			goto fail;
		}

		shape = new Py_ssize_t[2];
		shape[0]=feat_high-feat_low;
		shape[1]=vec_high-vec_low;

		strides=new Py_ssize_t[2];
		strides[0]=sizeof( type_name );
		strides[1]=sizeof( type_name )*num_feat;

		data+=feat_low*strides[0]+vec_low*strides[1];

		// not slice item should give vector or scalar
		if (type_item1==1)
		{
			// transpose
			shape++;
			strides++;
			ndims--;
		}
		if (type_item2==1)
		{
			ndims--;
		}

		if (ndims==0 && get_scalar)
		{
			ret=(PyObject *) PyArray_Scalar(data, descr, (PyObject *) self);
		}
		else
		{
			ret=(PyObject *) PyArray_NewFromDescr(&PyArray_Type, descr,
					ndims, shape,
					strides, data,
					NPY_FARRAY | NPY_WRITEABLE,
					(PyObject *) self);
		}

		if (ret==NULL)
		{
			// TODO error message
			goto fail;
		}

		Py_INCREF(self);
		return ret;
	}
	else if (PySlice_Check(key) || PyInt_Check(key) || PyArray_IsScalar(key, Integer) ||
       		PyLong_Check(key) || (PyIndex_Check(key) && !PySequence_Check(key)))
	{
		int item_type;
		item_type=parse_tuple_item(key, num_feat,
							&feat_low, &feat_high,
							&feat_step, &feat_slicelength);

		switch (item_type)
		{
		case 1:
			return class_name ## _getitem(self, feat_low);
			break;
		case 2:
			return class_name ## _getslice(self, feat_low, feat_high);
			break;
		default:
			goto fail;
		}
	}

fail:
	return NULL;
}

/* used for numpy's style slicing */
static int class_name ## _setsubscript(PyObject *self, PyObject *key, PyObject* v)
{
	PyArrayObject* tmp=NULL;
	int ret=0;

	if (v==NULL)
	{
		// TODO error message
		goto fail;
	}

	tmp = (PyArrayObject *) class_name ## _getsubscript(self, key, false);
	if(tmp == NULL)
	{
		goto fail;
	}
	ret = PyArray_CopyObject(tmp, v);
	Py_DECREF(tmp);
	return ret;

fail:
	return -1;
}

NUMERIC_DENSEFEATURES(class_name, type_name, format_str, add, +)
NUMERIC_DENSEFEATURES(class_name, type_name, format_str, sub, -)
NUMERIC_DENSEFEATURES(class_name, type_name, format_str, mul, *)

static long class_name ## _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_NEWBUFFER | Py_TPFLAGS_BASETYPE;
%}

%init
%{
SwigPyBuiltin__shogun__CDenseFeaturesT_ ## type_name ## _t_type.ht_type.tp_flags = class_name ## _flags;
%}

%feature("python:bf_getbuffer") CDenseFeatures< type_name > #class_name "_getbuffer"
%feature("python:bf_releasebuffer") CDenseFeatures< type_name > #class_name "_releasebuffer"

%feature("python:nb_inplace_add") CDenseFeatures< type_name > #class_name "_inplaceadd"
%feature("python:nb_inplace_subtract") CDenseFeatures< type_name > #class_name "_inplacesub"
%feature("python:nb_inplace_multiply") CDenseFeatures< type_name > #class_name "_inplacemul"

%feature("python:sq_item") CDenseFeatures< type_name > #class_name "_getitem"
%feature("python:sq_ass_item") CDenseFeatures< type_name > #class_name "_setitem"
%feature("python:sq_slice") CDenseFeatures< type_name > #class_name "_getslice"
%feature("python:sq_ass_slice") CDenseFeatures< type_name > #class_name "_setslice"

%feature("python:mp_subscript") CDenseFeatures< type_name > #class_name "_getsubscript"
%feature("python:mp_ass_subscript") CDenseFeatures< type_name > #class_name "_setsubscript"

%enddef /* PYPROTO_DENSEFEATURES */
#endif /* SWIG_PYTHON */
