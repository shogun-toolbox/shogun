/* Buffer protocol stuff for DenseFeatures */
%define BUFFER_DENSEFEATURES(class_name, type_name, type, format_str)

%wrapper 
%{

	static int class_name ## _getbuffer(PyObject *exporter, Py_buffer *view, int flags)
	{
		CDenseFeatures< type > * self = (CDenseFeatures< type > *) 0;
		void *argp1 = 0 ;
		int res1 = 0 ;

		int num_feat = 0, num_vec = 0;

		Py_ssize_t* shape;
		Py_ssize_t* stride;

		static char* format = format_str;

		res1 = SWIG_ConvertPtr(exporter, &argp1, SWIG_TypeQuery("shogun::CDenseFeatures<type>"), 0 |  0 );
		if (!SWIG_IsOK(res1)) 
		{
			SWIG_exception_fail(SWIG_ArgError(res1), "in method '" " class_name _getbuffer" "', argument " "1"" of type '" "CDenseFeatures< type > *""'"); 
		}

		self = reinterpret_cast< CDenseFeatures < type > * >(argp1);

		view->buf = self->get_feature_matrix(num_feat, num_vec);

		shape = new Py_ssize_t[2];
		shape[0] = num_feat;
		shape[1] = num_vec;

		stride = new Py_ssize_t[2];
		stride[0] = sizeof( type );
		stride[1] = sizeof( type ) * num_feat;
		
		view->len = shape[0]*stride[0];
		view->itemsize = stride[0];
		view->readonly = 0;

		/* TODO fix warnings related to const char* -> char* */
		view->format = format;		
		view->ndim = 2;
		view->shape = shape;
		view->strides = stride;
		view->suboffsets = NULL;
		view->internal = NULL;

		view->obj = (PyObject*) exporter;
		Py_INCREF(exporter);

		return 0;

fail:
		return -1;
	}

	static void class_name ## _releasebuffer(PyObject *exporter, Py_buffer *view)
	{
		if(view->shape != NULL)
			delete[] view->shape;
		
		if(view->strides != NULL)
			delete[] view->strides;
	}

	static long class_name ## _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_NEWBUFFER | Py_TPFLAGS_BASETYPE;
%}

%init
%{
	/* TODO less "hacked" */
	SwigPyBuiltin__shogun__CDenseFeaturesT_ ## type_name ## _t_type.ht_type.tp_flags = class_name ## _flags;
%}

%feature("python:bf_getbuffer") CDenseFeatures< type_name > #class_name "_getbuffer"
%feature("python:bf_releasebuffer") CDenseFeatures< type_name > #class_name "_releasebuffer"

%enddef
