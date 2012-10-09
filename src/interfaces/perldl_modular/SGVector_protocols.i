/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */
//PTZ121008can we use Math::GSL::Vector ??!!

#ifdef SWIGPERL

%include "protocols_helper.i"

//PTZ121002 need it for is_piddle and so
 //%include "swig_typemaps.i"

/* Numeric operators for SGVector */

%define NUMERIC_SGVECTOR(class_name, type_name, format_str, operator_name, operator)
 //PTZ121002 STILL do not use this...prefere PDL
XS(class_name ## _inplace ## operator_name ##)
{
  int argvi = 0;
  pdl* self; 
  dXSARGS;

  STMT_START {
    SV* const xsub_tmp_sv = ST(0);
    SvGETMAGIC(xsub_tmp_sv);
    if(is_piddle(xsub_tmp_sv)) {
      self = SvPDLV(xsub_tmp_sv);
    } else {
      Perl_croak
	(aTHX_ "%s: %s is not a piddle reference"
	 , "shogun::SGVector<type_name>:: class_name ## _inplace ## operator_name ##"
	 , "s");
    }
  } STMT_END;
  {
    dSP;
    I32 ax;
    int count;    
    ENTER;
    SAVETMPS;    
    PUSHMARK(SP);
    XPUSHs(sv_2mortal(ST(0)));
    //XPUSHs(sv_2mortal(ST(1)));
    //XPUSHs(ST(0));
    XPUSHs(ST(1));
    PUTBACK;

    XS_PDL_setflag(self->state, PDL_INPLACE, 1);

    count = call_pv(## operator ##, G_DISCARD);      
    SPAGAIN;
    SP -= count;
    ax = (SP - PL_stack_base) + 1;//used 2
    if (count != 1)
      SWIG_croak("Big trouble\n");
    PUTBACK;
    FREETMPS;
    LEAVE;
  }
  ST(argvi) = sv_2mortal(ST(0)); //macro "sv_newmortal" passed 1 arguments, but takes just 0
  argvi++;
  XSRETURN(argvi);

 fail:
  XSRETURN_UNDEF;
}
%enddef // NUMERIC_SGVECTOR


#if 0

 //inplace example

$a = pdl(0.3);
$a->inplace->sinh;
ok( tapprox($a, pdl(0.3045)) );

    //printf ("%d + %d = %d\n", a, b, SvIV(ST(0)));
    //printf ("%d - %d = %d\n", a, b, SvIV(ST(1)));

  //SV* resultobj = 0;
  //SGVector< type_name >* arg1 = (SGVector< type_name >*) 0; // self in c++ repr
  pdl* arg1 = 0; // self in c++ repr
  void* argp1 = 0; // pointer to self
  int res1 = 0; // result for self's casting

  //SGVector< type_name >* arg2 = (SGVector< type_name >*) 0;
  pdl* arg2;
  void *argp2 = 0 ;
  int res2 = 0 ;
  //pdl* internal_data;

#endif

#if 0
  //done ST(0-1) shall  already be a RV->PDSV
  res1 = SWIG_ConvertPtr(ST(0), &argp1, SWIG_TypeQuery("shogun::pdl"),  0 |  0 );
  if (!SWIG_IsOK(res1)) {
      SWIG_exception_fail(SWIG_ArgError(res1)
			  , "in method '" "inplace_#operator_name" "', argument " "1"" of type '" "pdl *""'");
    }
#endif
#if 0
  res2 = SWIG_ConvertPtr(ST(1), &argp2, SWIG_TypeQuery("shogun::pdl"), SWIG_POINTER_DISOWN |  0 );
  if (!SWIG_IsOK(res2)) {
    SWIG_exception_fail(SWIG_ArgError(res2)
			, "in method '" "inplace_#operator_name" "', argument " "2"" of type '" "SGVector<type_name> *""'");
  }
  //arg1 = reinterpret_cast< SGVector< type_name >* >(argp1);

//else

  ///convertions in swig_typemap.1

  STMT_START {
    SV* const xsub_tmp_sv = ST(1);
    SvGETMAGIC(xsub_tmp_sv);
    if (SvROK(xsub_tmp_sv) && SvTYPE(SvRV(xsub_tmp_sv)) == SVt_PVAV){
      AV* my_av = (AV*)SvRV(xsub_tmp_sv);

#if 0
  //TODO:PTZ120925 self shall be there already!!
  //get array...  //splice self,0,arg1->vlen)
  //AV*  av_make(I32 num, SV **ptr);
  //AV* internal_data = NULL;
  //internal_data = PySequence_GetSlice(self, 0, arg1->vlen);
  //http://search.cpan.org/~chm/PDL-2.4.11/Basic/SourceFilter/NiceSlice.pm

      AV* internal_data_av = av_make(arg1->vlen, self);
      if(av_len(internal_data) != arg1->vlen) {
	SWIG_exception_fail(SWIG_ERROR, "bad AV length");
      }

  //TODO:PTZ120925 never seen perl equivalent!! them map with op.
  //
  /*PyObject* PyNumber_InPlaceSubtract(PyObject *o1, PyObject *o2)
    Return value: New reference.
    Returns the result of subtracting o2 from o1, or NULL on failure. The operation is done in-place when o1 supports it.
    This is the equivalent of the Python statement o1 -= o2.
    **oops!!!
    opcode.h
    Perl_pp_
    i_add -> +=
    i_substract -=
    i_multiply *=

Perl_ppaddr_t PL_ppaddr[] 

  */

  // do a map...I am too leazy to put it in an op:
  // map($_->[0] $op $_->[1], 
  I32 flags = 0;
  for(int i = 0; i < arg1->vlen && i < arg2->vlen); i++) {
  //SV** bp = av_fetch(arg1p, i, 0);
  //SV** ap = av_fetch(arg2p, i, 0);
  // Perl_
  //Perl_pp_ ## operator ## (*ap, *bp);
  //OP * pp_op = newOP(, flags); 
  
  }
#endif

    vector_to_pdl(&xsub_tmp_sv, )
      //make piddle

    } elseif(is_piddle(xsub_tmp_sv)) {
      arg1 = SvPDLV(xsub_tmp_sv);

    } else {

    //Perl_croak
    SWIG_croak
      (aTHX_ "%s: %s is not a ARRAY reference"
       , "shogun::SGVector<type_name>:: class_name ## _inplace ## operator_name ##"
       , "s");
  }
} STMT_END;
//arg2 = reinterpret_cast< SGVector< type_name >* >(argp2);

  //PTZ121002
  // internal_data = PySequence_GetSlice(self, 0, arg1->vlen);
  //internal_data = slice('0:'. arg1->nelem())
  // can we use 
  //PyNumber_InPlace ## operator ## (internal_data, o2);
  //pdl->set_inline(1);
  //use affine+inline in the first
  //really SGVector< type_name > shall be created inside pdl*, then use perl-dl/api to grok it.
  //warn("so this inline is optional for now...");
  //not really!!
  //resultobj=self;
  //INCREF(resultobj);
  //return resultobj;
#endif

/* Perl protocols for SGVector */

 ///usr/src/shogun/src/interfaces/modular/Library.i:285:
 //PROTOCOLS_SGVECTOR(ShortRealVector, float32_t, "f\0", NPY_FLOAT32)
 ///usr/src/shogun/src/interfaces/modular/Library.i:290:	
 //PROTOCOLS_SGVECTOR(RealVector, float64_t, "d\0", NPY_FLOAT64)


%define PROTOCOLS_SGVECTOR(class_name, type_name, format_str, typecode)
%wrapper
%{
/* used by PDL_GetBuffer !!!*/
#if 0
  //view is a piddle?
  HV* pdl_hv;
  STMT_START {
    SV* const xsub_tmp_sv = view;
    SvGETMAGIC(xsub_tmp_sv);
    if (SvROK(xsub_tmp_sv) && SvTYPE(SvRV(xsub_tmp_sv)) == SVt_PVHV){
      pdl_hv = (HV*)SvRV(xsub_tmp_sv);
    }
    else{
      croak(aTHX_ "%s: %s is not a HASH reference",
		 "class_name ## _getbuffer ",
		 "view");
    }
  } STMT_END;
#endif
#if defined __who_cares_in_perl
  if((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS
      || ((flags & PyBUF_STRIDES) != PyBUF_STRIDES
	  && (flags & PyBUF_ND) == PyBUF_ND)
      )
    {
//ERRSV SWIG_ArgError(NULL)
      SWIG_exception_fail(ERRSV, "class_name is not C-contiguous");
    }
#endif
#if 0 // hash it
  //Py_buffer mess... why one would?
  view->buf = info->buf.vector;
  view->ndim = 1;
  view->format = (char*) format_str;
  view->itemsize = sizeof( type_name );
  view->len= shape[0] * view->itemsize;
  view->shape = shape;
  view->strides = strides;
  view->readonly = 0;
  view->suboffsets = NULL;
  view->internal = (void*) info;
  view->obj= SvREFCNT_inc(self);
#endif
#if 0
    //if(pdl_view->hdrsv == NULL) {
    //  pdl_view->hdrsv =  &PL_sv_undef; /*(void*) newSViv(0);*/
    //}
    /* Throw an error if we're not either undef or hash */
    if ( (info_iv != &PL_sv_undef && h != NULL) &&
	 ( !SvROK(h) || SvTYPE(SvRV(h)) != SVt_PVHV )
	 )
      croak("Not a HASH reference");
    /* Clear the old header */
    SvREFCNT_dec(pdl_view->hdrsv);
    /* Put the new header (or undef) in place */
    if(info_iv == &PL_sv_undef || info_iv == NULL) {
      pdl_view->hdrsv = NULL;
    } else {
      pdl_view->hdrsv = (void*) newRV((SV*)info_iv);
    }  
  //SetSV_PDL (sv, pdl_view);
  //pdl_makescratchhash(pdl *ret,double data, int datatype);
  //hv_store(hash, "Dims", strlen("Dims"), newRV( (SV*) array), 0 );
  //SetSV_PDL(sv, pdl_view);
  //pdl_view = pdl_null();
  //SvREFCNT_dec(pdl_view->hdrsv); //info..
  //deref first...
  //pdl_view->hdrsv = NULL;

#endif
  //set header... with info
  //ret = INT2PTR(pdl *, SvIV(sv2));
  //HV* view_hv = SvRV(pdl_view->sv);
  //hv_store(hdrsv_hv, "info", strlen("info"), newRV( (SV*) array), 0 );
  //sethdr(pdl_view, info_pv);

static int class_name ## _getbuffer(SV* self, pdl* pdl_view, I32 flags)
{
  SGVector< type_name >* arg1 = (SGVector< type_name >*) 0; // self in c++ repr
  void* argp1 = 0; // pointer to self
  int res1 = 0; // result for self's casting
  
  int num_labels = 0;
  static char* format=(char *) format_str; //or "f\0";
  //dXSARGS;ST(0)
  res1 = SWIG_ConvertPtr(self, &argp1, SWIG_TypeQuery("shogun::SGVector<type_name>"), 0 |  0 );
  if (!SWIG_IsOK(res1)) {
    //SWIG_exception_fail(SWIG_ArgError(res1), "in method '" "getbuffer" "', argument " "1"" of type '" "SGVector<type_name> *""'");
  }
  arg1 = reinterpret_cast< SGVector< type_name >* >(argp1);

  //PTZ120925 sounds like packing...
  //info;
  buffer_vector_ ## type_name ## _info* info = new buffer_vector_ ## type_name ## _info;
  info->buf = *arg1;
  num_labels = arg1->vlen;

  STRLEN* shape = new STRLEN[1];
  shape[0] = num_labels;
  info->shape = shape;

  STRLEN* strides = new STRLEN[1];
  strides[0] = sizeof( type_name );  
  info->strides = strides;

  { //SWIG_ConvertPtr(obj, pp, type, flags) 
    //IV* info_iv = PTR2IV(info);
    SV* info_sv;
    //nop, the other way roundflags...(SWIG_SHADOW | SWIG_POINTER_OWN)
    SWIG_MakePtr(info_sv, (void*) info, SWIG_TypeQuery("shogun::buffer_vector_ ## type_name ## _info"), 0);
    //if(!SWIG_IsOK(res)) {
    //	SWIG_exception_fail(SWIG_ArgError(res), "pointert convertionshogun::buffer_vector_ ## type_name ## _info");
    //}

    if((pdl_view->hdrsv==NULL) || (pdl_view->hdrsv == &PL_sv_undef)) {
      pdl_view->hdrsv = (void*) newRV_noinc( (SV*)newHV());
    }
    STMT_START {
      SV* const xsub_tmp_sv = (SV*) pdl_view->hdrsv;
      SvGETMAGIC(xsub_tmp_sv);
      if (SvROK(xsub_tmp_sv) && SvTYPE(SvRV(xsub_tmp_sv)) == SVt_PVHV){
	HV* pdl_hv = (HV*)SvRV(xsub_tmp_sv);	    
	hv_store(pdl_hv, "info", strlen("info"), info_sv, 0);
      } else {
	//warm("not storing info"); // has not been declared
      }
    } STMT_END;
  }


    {
      PDL_Long pdl_dims[1] = {0};

      pdl_makescratchhash(pdl_view, 0.0, PDL_B);
      pdl_setdims(pdl_view, pdl_dims, 1);
      //etc...?
    }
  return 0;
fail:
  return -1;
}

/* used by PyBuffer_Release */
//(PyObject *self, Py_buffer *view)
 static void class_name ## _releasebuffer(SV* self, pdl* pdl_view)
 {

      SV* const xsub_tmp_sv = (SV*) pdl_view->hdrsv;
      SvGETMAGIC(xsub_tmp_sv);
      if (SvROK(xsub_tmp_sv) && SvTYPE(SvRV(xsub_tmp_sv)) == SVt_PVHV){
	HV* pdl_hv = (HV*)SvRV(xsub_tmp_sv);	    
	void* info_p;
	SV** info_rv_p = hv_fetch(pdl_hv, "info", strlen("info"), 0);
	int res = SWIG_ConvertPtr(*info_rv_p, &info_p, SWIG_TypeQuery("shogun::buffer_vector_ ## type_name ## _info"), 0 |  0 );
	if(!SWIG_IsOK(res))
	{
	  //SWIG_exception_fail(SWIG_ArgError(res), "could not retrieve info in class_name ## _releasebuffer");
	}
	buffer_vector_ ## type_name ## _info* temp = reinterpret_cast< buffer_vector_ ## type_name ## _info* >(info_p);
   //view is a *Py_Buffer??
	{
	  //temp = (buffer_vector_ ## type_name ## _info*) view->internal;
	  if(temp != NULL) {
	    if (temp->shape != NULL)
	      delete[] temp->shape;
	    if (temp->strides != NULL)
	      delete[] temp->strides;
	    temp->buf = SGVector< type_name >();
	    delete temp;
	  }       
	}	
	//STMT_START {
	//} STMT_END;
      }
#if 0
   buffer_vector_ ## type_name ## _info* temp = NULL;
   if (view->obj != NULL && view->internal!=NULL)
     {
       temp = (buffer_vector_ ## type_name ## _info*) view->internal;
       if(temp != NULL) {
	 if (temp->shape != NULL)
	   delete[] temp->shape;       
	 if (temp->strides != NULL)
	   delete[] temp->strides;
	 temp->buf = SGVector< type_name >();
	 delete temp;
       }       
     }
#endif
 }

/* used by PySequence_GetItem */
 static SV* class_name ## _getitem(SV* self, size_t idx, bool get_scalar = true)
 {
   //PTZ121004 not ready then
 fail:
   return NULL;
 }

#if defined PTZ121001_not_ready_yet_
  //check PDL get_slice
  // not needed?
  SGVector< type_name >* arg1=(SGVector< type_name >*) 0; // self in c++ repr
  void* argp1=0; // pointer to self
  int res1=0; // result for self's casting

  char* data=0; // internal data of self
  int vlen=0;

  SGVector< type_name > temp;

  size_t* shape;
  size_t* strides;

  SV* ret;

  PyArray_Descr* descr=PyArray_DescrFromType(typecode);

  res1 = SWIG_ConvertPtr(ST(0), &argp1, SWIG_TypeQuery("shogun::SGVector<type_name>"), 0 |  0 );
  if (!SWIG_IsOK(res1))
    {
      SWIG_exception_fail(SWIG_ArgError(res1),
			  "in method '" "getitem" "', argument " "1"" of type '" "SGVector<type_name> *""'");
    }
  arg1=reinterpret_cast< SGVector< type_name >* >(argp1);
	
  temp = *arg1;
  vlen = arg1->vlen;

  data = (char*) temp.vector;

  idx = get_idx_in_bounds(idx, vlen);
  if (idx < 0)
    {
      goto fail;
    }

  data += idx * sizeof( type_name );

  shape = new size_t[1];
  shape[0] = 1;

  strides = new size_t[1];
  strides[0] = sizeof( type_name );

  if (get_scalar)
    {
      ret=(PyObject *) PyArray_Scalar(data, descr, (PyObject *) self);
    }
  else
    {
      ret=(PyObject *) PyArray_NewFromDescr(&PyArray_Type, descr,
					    0, shape,
					    strides, data,
					    NPY_FARRAY | NPY_WRITEABLE,
					    (PyObject *) self);
    }

  if (ret==NULL)
    goto fail;

  Py_INCREF(self);
  return ret;
#endif


/* used by PySequence_SetItem */

#if 0
XS(XS_PDL_set_data_by_offset)
{
#ifdef dVAR
    dVAR; dXSARGS;
#else
    dXSARGS;
#endif
    if (items != 3)
       croak_xs_usage(cv,  "it, orig, offset");
    {
	pdl *	it = SvPDLV(ST(0));
	pdl *	orig = SvPDLV(ST(1));
	STRLEN	offset = (STRLEN)SvUV(ST(2));
	int	RETVAL;
	dXSTARG;
#line 557 "Core.xs"
              pdl_freedata(it);
              it->data = ((char *) orig->data) + offset;
	      it->datasv = orig->sv;
              SvREFCNT_inc(it->datasv);
              it->state |= PDL_DONTTOUCHDATA | PDL_ALLOCATED;
              RETVAL = 1;
#line 943 "Core.c"
	XSprePUSH; PUSHi((IV)RETVAL);
    }
    XSRETURN(1);
}

  PyArrayObject* tmp=NULL;
  int ret = 0;
  if (v==NULL)
    {
      // TODO error message
      goto fail;
    }
  tmp = (PyArrayObject *) class_name ## _getitem(self, idx, false);
  if(tmp == NULL)
    {
      goto fail;
    }
  ret=PyArray_CopyObject(tmp, v);
  Py_DECREF(tmp);
#endif

static int class_name ## _setitem(SV* self, size_t idx, SV* v)
{
  int ret = 0;
  //get an object in place... idx is vaffine offset?
  SV* tmp;
  if (v == NULL) {
    // TODO error message
    goto fail;
  }
  tmp = class_name ## _getitem(self, idx, false); //AV
  if(tmp == NULL) {
    goto fail;
  }
  //PTZ121004 todo...
  //put SV v instead?
  //ret = PDL->set_data_by_offset(self, v, idx);
  {
    dSP;
    I32 ax;
    int count;    
    ENTER;
    SAVETMPS;    
    PUSHMARK(SP);
    XPUSHs(sv_2mortal(self));
    XPUSHs(sv_2mortal(v));
    XPUSHs(sv_2mortal(newSViv(idx)));    
    PUTBACK;
    count = call_pv("PDL::set_data_by_offset", G_DISCARD);      
    SPAGAIN;
    SP -= count;
    ax = (SP - PL_stack_base) + 1;//used 2
    if (count != 1)
      SWIG_croak("Big trouble\n");
    PUTBACK;
    FREETMPS;
    LEAVE;
  }
  return ret;
fail:
  return -1;
}


/* used by PySequence_GetSlice */
static SV* class_name ## _getslice(SV *self, size_t ilow, size_t ihigh)
{
  pdl*	it = SvPDLV(self);

#if defined PTZ121001_not_yet
	SGVector< type_name >* arg1=(SGVector< type_name >*) 0; // self in c++ repr
	void* argp1=0; // pointer to self
	int res1=0 ; // result for self's casting

	int vlen=0;
	char* data=0; // internal data of self

	SGVector< type_name > temp;

	Py_ssize_t* shape;
	Py_ssize_t* strides;

	PyArrayObject* ret;
	PyArray_Descr* descr=PyArray_DescrFromType(typecode);

	res1=SWIG_ConvertPtr(self, &argp1, SWIG_TypeQuery("shogun::SGVector<type_name>*"), 0 |  0 );
	if (!SWIG_IsOK(res1))
	{
		SWIG_exception_fail(SWIG_ArgError(res1),
					"in method '" "slice" "', argument " "1"" of type '" "SGVector<type_name> *""'");
	}

	arg1=reinterpret_cast< SGVector< type_name >* >(argp1);

	temp=*arg1;
	vlen=arg1->vlen;

	data=(char*) temp.vector;

	get_slice_in_bounds(&ilow, &ihigh, vlen);
	if (ilow < ihigh)
	{
		data+=ilow * sizeof( type_name );
	}

	shape=new Py_ssize_t[1];
	shape[0]=ihigh - ilow;

	strides=new Py_ssize_t[1];
	strides[0]=sizeof( type_name );

	ret=(PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type, descr,
					1, shape,
 					strides, data,
 					NPY_FARRAY | NPY_WRITEABLE,
 					(PyObject *) self);
	if (ret==NULL)
		goto fail;

	Py_INCREF(self);
	return (PyObject *) ret;

#endif

	//fail:
	return NULL;
}

/* used by PySequence_SetSlice */
static int class_name ## _setslice(SV *self, size_t ilow, size_t ihigh, SV* v)
{
  SV* tmp = NULL;
  int ret = 0;
  if(v == NULL) {
    // TODO error message
    goto fail;
  }
  return ret;
 fail:
  return -1;
}
#if defined PTZ121001_not_yet_
	tmp=(PyArrayObject *) class_name ## _getslice(self, ilow, ihigh);
	if(tmp==NULL)
	{
		goto fail;
	}
	ret = PyArray_CopyObject(tmp, v);
	Py_DECREF(tmp);

#endif



//PTZ120925 operators... //do not care about type_name...PDL does it.
NUMERIC_SGVECTOR(class_name, type_name, format_str,  XS_PDL_plus, "PDL::plus")
//NUMERIC_SGVECTOR(class_name, type_name, format_str, sub, OP_I_SUBTRACT)
//NUMERIC_SGVECTOR(class_name, type_name, format_str, mul, OP_I_MULTIPLY)


#if 0
//here all the operators in PDL!!!
//NUMERIC_SGVECTOR(class_name, type_name, format_str, add, OP_I_ADD)
//NUMERIC_SGVECTOR(class_name, type_name, format_str, sub, OP_I_SUBTRACT)
//NUMERIC_SGVECTOR(class_name, type_name, format_str, mul, OP_I_MULTIPLY)
/*
        (void)newXSproto_portable("PDL::Ops::set_debugging", XS_PDL__Ops_set_debugging, file, "$");
        (void)newXSproto_portable("PDL::Ops::set_boundscheck", XS_PDL__Ops_set_boundscheck, file, "$");
        (void)newXSproto_portable("PDL::plus", XS_PDL_plus, file, ";@");
        (void)newXSproto_portable("PDL::mult", XS_PDL_mult, file, ";@");
        (void)newXSproto_portable("PDL::minus", XS_PDL_minus, file, ";@");
        (void)newXSproto_portable("PDL::divide", XS_PDL_divide, file, ";@");
        (void)newXSproto_portable("PDL::gt", XS_PDL_gt, file, ";@");
        (void)newXSproto_portable("PDL::lt", XS_PDL_lt, file, ";@");
        (void)newXSproto_portable("PDL::le", XS_PDL_le, file, ";@");
        (void)newXSproto_portable("PDL::ge", XS_PDL_ge, file, ";@");
        (void)newXSproto_portable("PDL::eq", XS_PDL_eq, file, ";@");
        (void)newXSproto_portable("PDL::ne", XS_PDL_ne, file, ";@");
        (void)newXSproto_portable("PDL::shiftleft", XS_PDL_shiftleft, file, ";@");
        (void)newXSproto_portable("PDL::shiftright", XS_PDL_shiftright, file, ";@");
        (void)newXSproto_portable("PDL::or2", XS_PDL_or2, file, ";@");
        (void)newXSproto_portable("PDL::and2", XS_PDL_and2, file, ";@");
        (void)newXSproto_portable("PDL::xor", XS_PDL_xor, file, ";@");
        (void)newXSproto_portable("PDL::bitnot", XS_PDL_bitnot, file, ";@");
        (void)newXSproto_portable("PDL::power", XS_PDL_power, file, ";@");
        (void)newXSproto_portable("PDL::atan2", XS_PDL_atan2, file, ";@");
        (void)newXSproto_portable("PDL::modulo", XS_PDL_modulo, file, ";@");
        (void)newXSproto_portable("PDL::spaceship", XS_PDL_spaceship, file, ";@");
        (void)newXSproto_portable("PDL::sqrt", XS_PDL_sqrt, file, ";@");
        (void)newXSproto_portable("PDL::abs", XS_PDL_abs, file, ";@");
        (void)newXSproto_portable("PDL::sin", XS_PDL_sin, file, ";@");
        (void)newXSproto_portable("PDL::cos", XS_PDL_cos, file, ";@");
        (void)newXSproto_portable("PDL::not", XS_PDL_not, file, ";@");
        (void)newXSproto_portable("PDL::exp", XS_PDL_exp, file, ";@");
        (void)newXSproto_portable("PDL::log", XS_PDL_log, file, ";@");
        (void)newXSproto_portable("PDL::_log10_int", XS_PDL__log10_int, file, "$$");
        (void)newXSproto_portable("PDL::assgn", XS_PDL_assgn, file, ";@");
*/
#endif

//static long class_name ## _flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_NEWBUFFER | Py_TPFLAGS_BASETYPE;
//PTZ120925 flags to do....!
static long class_name ## _flags = 0;

%}

%init
%{
  //SwigPerlBuiltin__shogun__SGVectorT_ ## type_name ## _t_type.ht_type.tp_flags = class_name ## _flags;
  //  SWIGTYPE_p_shogun__SGVectorT_ ## type_name ## _t_type.ht_type.tp_flags = class_name ## _flags;
  
%}

%feature("perl:bf_getbuffer") SGVector< type_name > #class_name "_getbuffer"
%feature("perl:bf_releasebuffer") SGVector< type_name > #class_name "_releasebuffer"

%feature("perl:nb_inplace_add") SGVector< type_name > #class_name "_inplaceadd"
%feature("perl:nb_inplace_subtract") SGVector< type_name > #class_name "_inplacesub"
%feature("perl:nb_inplace_multiply") SGVector< type_name > #class_name "_inplacemul"

%feature("perl:sq_item") SGVector< type_name > #class_name "_getitem"
%feature("perl:sq_ass_item") SGVector< type_name > #class_name "_setitem"
%feature("perl:sq_slice") SGVector< type_name > #class_name "_getslice"
%feature("perl:sq_ass_slice") SGVector< type_name > #class_name "_setslice"

%enddef /* PROTOCOLS_SGVECTOR */
#else

#define PROTOCOLS_SGVECTOR(class_name, type_name, format_str, typecode)
#endif /* SWIG_PERL */
