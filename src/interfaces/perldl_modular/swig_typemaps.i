/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This code is inspired by the perl data language typemaps
 *
 * Written (W) 2012 Christian Montanari
 */

%include "DenseFeatures_protocols.i"
%include "CustomKernel_protocols.i"
%include "DenseLabels_protocols.i"
%include "SGVector_protocols.i"

#ifdef HAVE_PDL

%include pdl.i

%{

#ifndef SWIG_FILE_WITH_INIT
#  define NO_IMPORT_ARRAY
#else
  void import_array() {}
#endif

#include <shogun/lib/DataType.h>

/* Functions to extract array attributes.
 */

  //pdl* pdl_from_array(AV* av, AV* dims, int type, pdl* p)


/* Given a SV, return a string describing its type.
 */
static const char* typecode_string(SV* a) {
  if(!SvOK(a)) return "C NULL value";
  if (SvVOK(a)) return "v-string";
  if (SvIOK(a)) return "int";
  if (SvNOK(a)) return "double";
  //if (SvOK(a)) return "array";
  if (SvROK(a) && ((SvTYPE(SvRV(a)) == SVt_PVMG) || (SvTYPE(SvRV(a)) == SVt_PVHV)))
    //{(SvROK(a) && SvTYPE(SvRV(a)) == SVt_PDLV))
    return "piddle";
  //TODO::PTZ120927, I am sure I can do better!!!, with returning the name of obj!
  //look at swig_.cxx
#if 0
  if (py_obj == NULL          ) return "C NULL value";
  if (PyCallable_Check(py_obj)) return "callable";
  if (PyUnicode_Check( py_obj)) return "unicode";
  if (PyString_Check(  py_obj)) return "string";
  if (PyLong_Check(    py_obj)) return "int";
  if (PyInt_Check(     py_obj)) return "int";
  if (PyFloat_Check(   py_obj)) return "float";
  if (PyDict_Check(    py_obj)) return "dict";
  if (PyList_Check(    py_obj)) return "list";
  if (PyTuple_Check(   py_obj)) return "tuple";
  if (PyModule_Check(  py_obj)) return "module";
  if (PyFile_Check(    py_obj)) return "file";
  if (PyInstance_Check(py_obj)) return "instance";
#endif
  return "unknown type";
}

static const char* typecode_string(int typecode) {
    const char* type_names[24] = {"bool","byte","unsigned byte","short",
        "unsigned short","int","unsigned int","long",
        "unsigned long","long long", "unsigned long long",
        "float","double","long double",
        "complex float","complex double","complex long double",
        "object","string","unicode","void", "piddle","notype","char"};
    const char* user_def="user defined";
    if (typecode > 24)
        return user_def;
    else
        return type_names[typecode];
}

//PTZ121010 doesnot think we need this since already called in the swig interface...
//const class->get_copy
static void* get_copy(void* src, size_t len)
{
    void* copy = SG_MALLOC(uint8_t, len);
    memcpy(copy, src, len);
    return copy;
}

/* =pod
 * 
 * Given a piddle, check to see if it is contiguous (packed).
 * If so,
 * return the input pointer and flag it as not a new object.  If it is
 * not contiguous, create a new AV using the original data,
 * flag it as a new object and return the pointer.
 * 
 * If array is NULL or dimensionality or typecode does not match
 * return NULL
 *
 * =cut
 */
static SV* make_contiguous
  (SV* ary
   , int* is_new_object
   , int dims
   , int typecode
   , bool force_copy = false
   )
{
    SV* array = NULL;
    pdl* my_pdl = NULL;

    /* mapping of SvPDLV */
    my_pdl = PDL->SvPDLV(ary);

    if(force_copy == true) {
      //const char* opt = "";
      //char* const opt = "";
      //char * opt = "";
      char opt[] = "";
      sv_setsv(array, pdl_copy(my_pdl, opt));
    } else {
      //pdl *pnew = pdl_hard_copy(my_pdl);
    }

    /* change dims... */
    //PDL_Long* my_dims_pdl = pdl_packdims(ary, ndims);
    //as in pdlapi.c
    //pdl_setdims(my_pdl, my_pdl->dims, dims);
    PDL->reallocdims(my_pdl, dims);
    //maybe pdl_setdims_careful(my_pdl);

    /* change type... */
    {
      pdl* old_pdl = my_pdl;
      my_pdl = pdl_get_convertedpdl(old_pdl, typecode);
      if(my_pdl != old_pdl) {
	pdl_destroy(old_pdl);
      }
    }
    PDL->make_physical(my_pdl);


    PDL->SetSV_PDL(array, my_pdl);

    if(array != ary) {
      *is_new_object = 1;
    } else  *is_new_object = 0;


    return array;

#if defined PTZ120927_here_yet
    //(SV*)SvRV(ary);


      //put dimension array pointers (RV_AV_IV into pdl_malloc() ...
      // not inc refrencing/ SvIV(*(av_fetch( array, i, 0 )))...
      //PDL_Long* ndims_pdl_p = pdl_packdims(ary, ndims);

    //sv_derived_from(sv, "PDL")
    //sv2 = (SV*) SvRV(sv);
    //ret = INT2PTR(pdl *, SvIV(sv2));
    //(PyArray_ISFARRAY)
    //SVavref(x)
    //(SvROK(x) && SvTYPE(SvRV(x))==SVt_PVAV)
    if((SvROK(ary) && SvTYPE(SvRV(ary)) == SVt_PVAV) && !force_copy)
    {
      array = ary;
      *is_new_object = 0;
    }
    else
    {
      //in /usr/lib/perl5/PDL/Core/pdlcore.h
      //forced copy,
      //PDL_Long *    pdl_packdims ( SV* sv, int*ndims );
 /* Pack dims[] into SV aref */
      //array = PyArray_FromAny((SV*)ary, NULL,0,0, NPY_FARRAY|NPY_ENSURECOPY, NULL);
      //SetSV_PDL(ary, p);

      //pdl* pdl_from_array(AV* av, AV* dims, int type, pdl* p);
      // array = 
      array_pdl = SvPDLV(ary);

      *is_new_object = 1;
    }
    if (!array)
    {
      barf("Object did convert to Empty object - not an Array ?");
        *is_new_object=0;
        return NULL;
    }

    if (!is_array(array))
    {
      barf("Object not a PDL Array");
      *is_new_object=0;
      return NULL;
    }


    if (dims!=-1 && array_dimensions(array) != dims)
    {
      warn("Array has wrong dimensionality, " 
                "expected a %dd-array, received a %dd-array", dims, array_dimensions(array));
        if (*is_new_object)
            Py_DECREF(array);
        *is_new_object=0;
        return NULL;
    }
    //TODO:PTZ120927 

    /*this works around a numpy oddity when LONG==INT32*/
    if ((array_type(array) != typecode) &&
        !(typecode==NPY_LONG && NPY_BITSOF_INT == NPY_BITSOF_LONG 
            && NPY_BITSOF_INT==32 && array_type(array)==NPY_INT))
    {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type = typecode_string(array_type(array));
        PyErr_Format(PyExc_TypeError, 
                "Array of type '%s' required.  Array of type '%s' given", 
                desired_type, actual_type);
        if (*is_new_object)
            Py_DECREF(array);
        *is_new_object=0;
        return NULL;
    }

    return array;
#endif

}


template <class type>
static bool vector_from_pdl(SGVector<type>& sg_vec, SV* obj, int typecode)
{
  pdl* it = SvPDLV(obj);
  it = pdl_get_convertedpdl(it, typecode);
  PDL->make_physical(it); /* Wasteful*/
  if(!PDL_ENSURE_ALLOCATED(it)) {
    warn("could not allocate PDL vector memory");
    return false;
  }
  //PTZ121008 still not sure of this one
  //SvREFCNT_inc(it->datasv);
  void* data = get_copy(PDL_REPRP(it), sizeof(type) * it->nvals);
  sg_vec = shogun::SGVector<type>((type*) data, it->dims[0]);

  //PTZ120927 shall I deference it? it will free it->data
  //pdl_destroy(it); //shall increment the vector? not here
    
  return true;
}



#if 0
    npy_intp dims= (npy_intp) sg_vec.vlen;
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);
    if (descr)
    {
        obj = PyArray_NewFromDescr(&PyArray_Type,
                descr, 1, &dims, NULL, copy, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((AV*) obj)->flags |= NPY_OWNDATA;
    }
    return descr!=NULL;

    shogun::SGVector< bool > *arg1 = (shogun::SGVector< bool > *) 0 ;
    bool arg2 ;
    void *argp1 = 0 ;
    int res1 = 0 ;
    bool val2 ;
    int ecode2 = 0 ;
    int argvi = 0;
    shogun::SGVector< shogun::index_t > result;
    dXSARGS;
    res1 = SWIG_ConvertPtr(ST(0), &argp1,SWIGTYPE_p_shogun__SGVectorT_bool_t, 0 |  0 );
    if (!SWIG_IsOK(res1)) {
      SWIG_exception_fail(SWIG_ArgError(res1), "in method '" "BoolVector_find" "', argument " "1"" of type '" "shogun::SGVector< bool > *""'"); 
    }
    arg1 = reinterpret_cast< shogun::SGVector< bool > * >(argp1);

//vector_to_pdl(ST(argvi), result, PDL_L))
//  shogun::SGVector< shogun::index_t > result;
//= pdl_malloc(sizeof(PDL_Long));
//int  = pdl_howbig(typecode); //pdl_from_array();

//that is bizare... this is used wrong way round... in wrapper...
      if (!vector_to_pdl(ST(argvi), result, PDL_L))
      SWIG_fail;
  //pdl_makescratchhash(it, 0.0, typecode);
  //pdl* it = PDL->SvPDLV ( rsv );

it->state |= PDL_DONTTOUCHDATA;

#endif

#if 0
  HV *bless_stash = 0;
 SV *parent = rsv;

 char *objname = "PDL";
       PUSHMARK(SP);
       XPUSHs(sv_2mortal(newSVpv(objname, 0)));
       PUTBACK;
       perl_call_method("initialize", G_SCALAR);
       SPAGAIN;
       SV *b_SV = POPs;
       PUTBACK;
       //it = PDL->SvPDLV(b_SV);
  if (bless_stash) rsv = sv_bless(rsv, bless_stash);


  //it->data = get_copy(sg_vec.vector, clen);
  //it->datasv = newSVpvn((char*)it->data, clen);
  //it->data =(void *) SvPV( it->datasv , clen );

#endif

//PTZ121005 not really ready, need to know what is given PDL or sg_vec
//PTZ121004 my understanding is sg_vec given, but why return it...?


template <class type>
static bool vector_to_pdl(SV* rsv, SGVector<type> sg_vec, int typecode)
{
  pdl* it = PDL->pdlnew();
  if(!it) {
    return false;
  }
  STRLEN clen = sizeof(type) * size_t(sg_vec.vlen);
  PDL_Long dims[1] = {size_t(sg_vec.vlen)};
  PDL->setdims(it, dims, 1);
  it->datatype = typecode;
  PDL->allocdata(it);
  void* data = PDL_REPRP(it);    
  if(!data) {
    PDL->destroy(it);
    return false;
  }
  memcpy((type*)data, sg_vec.vector, clen);
  PDL->SetSV_PDL(rsv, it);
  return true;
}


#if 0
    int is_new_object;
    SV* array = make_contiguous(obj, &is_new_object, 2,typecode, true);
    if (!array)
        return false;
    sg_matrix = shogun::SGMatrix<type>((type*) PyArray_BYTES(array),
            PyArray_DIM(array,0), PyArray_DIM(array,1), true);

    ((AV*) array)->flags &= (-1 ^ NPY_OWNDATA);
    Py_DECREF(array);
 
   if(it->ndims < 2) {
      warn("not a PDL matrix");
      return false;
    }
 
#endif
//PTZ121002 wont work as typemap AV* would not work well?
    // comparable to make_contiguous
    //type* m = (type*) it->data;
  //pdl *pdl_hard_copy(pdl *src) src to a new...
  //PTZ120928 shall I reference it? as it->data might be reuseable! by sg_matrix indeed
  //something like incREF(it->datasv);
  //also  make use of it->typecode and typecode, and type;
  //pdl *pdl_get_convertedpdl(pdl *old,int type) {    //it->datatype = typecode;
  //SvREFCNT_inc(it->datasv);
  // so it would need to e free later with shogun ?? or we cannot avoid the copy?
  // checkout the SG_MALLOC
  //PTZ121011 dims seems to be transposed from normal order

template <class type>
static bool matrix_from_pdl(SGMatrix<type>& sg_matrix, SV* obj, int typecode)
{
  pdl* it = if_piddle(obj);
  it = pdl_get_convertedpdl(it, typecode);
  PDL->make_physical(it); /* Wasteful but needed */
  if(!PDL_ENSURE_ALLOCATED(it)) {
    warn("could not allocate PDL (rectangular) matrix memory");
    return false;
  }
  void* data = get_copy(PDL_REPRP(it), sizeof(type) * it->nvals);
  sg_matrix = shogun::SGMatrix<type>((type*) data, it->dims[1], it->dims[0], true);    
  return true;
}


#if 0
    npy_intp dims[2]= {(npy_intp) sg_matrix.num_rows, (npy_intp) sg_matrix.num_cols };
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);

    if (descr)
    {
        void* copy=get_copy(sg_matrix.matrix, sizeof(type)*size_t(sg_matrix.num_rows)*size_t(sg_matrix.num_cols));
        obj = PyArray_NewFromDescr(&PyArray_Type,
            descr, 2, dims, NULL, (void*) copy, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((AV*) obj)->flags |= NPY_OWNDATA;
    }
    return descr != NULL;
#endif
#if 0
    pdl* it = pdl_new();
   //= pdl_malloc(sizeof(PDL_Long));
    pdl_makescratchhash(it, 0.0, typecode);
    pdl_setdims(it, d, 2);
    it->datatype = typecode;
    pdl_allocdata(it);
    //    pdl_make_physical(it);
 
    void* copy = get_copy(sg_matrix.matrix
			  , sizeof(type)
			  * size_t(sg_matrix.num_rows)
			  * size_t(sg_matrix.num_cols));
    free(it->data);
    it->data = copy;


    //PTZ120928unlikely... pdl_setav_
    //set an AV, then stuff it to te pdl...
    SV** sv_pl_p ;
    SV* dims_sv = newSV();
    (AV*)SvRV(dims_rv);



    STRLEN clen = sizeof(type)
      * size_t(sg_matrix.num_rows)
      * size_t(sg_matrix.num_cols);
    type* copy = (type*) get_copy(sg_matrix.matrix, clen);


    //from pdl_unpackdims

#endif
#if 0
    AV* dims_av = newAV();
    AV* data_av = newAV();
    AV* data_i_av;
    int ij = 0;
    int ndims = 2;
    PDL_Long dims[2] = {sg_matrix.num_cols, sg_matrix.num_rows};
    for(int i = 0; i < ndims; i++)
	av_store(dims_av, i, (SV*)newSViv((IV) dims[i]));
    for(int i = 0; i < sg_matrix.num_cols; i++) {
      data_i_av = newAV();
      for(int j = 0; j < sg_matrix.num_rows; j++) {
	av_store(data_i_av, j, newSVnv((NV) *(sg_matrix.matrix + ij)));
	ij++;
      }
      av_store(data_av, i, newRV((SV*) data_i_av));
    }
    pdl* it = pdl_from_array(data_av, dims_av, typecode, NULL);    
    PDL->SetSV_PDL(rsv, it);
    //PTZ121005assign sv_pdl to rsv???newRV((SV*) sv_pdl)
    //PTZ120928 free copy? check also nmap type of copy in Core.xs
    return true;
#endif
    //PTZ121010 
    //yet another way of filling a piddle using pdl_from_array()
    //it is inneficient... a memcopy shall be better used:
    // take vector_to_pdl as an example....
    //
    //pdl_unpackdims(dims_sv, d, 2)
    //sv_setpvn((SV*)data_pv, (const char*)copy, len);
    //PTR2UV(copy);
    //PTZ120928 very innefficient indeed, and not right, need to check SGMatrix<type>
    //PTZ120928 typecode need to be cast to perl types? type to IV,NV,PV?
    //AV*     av_make(I32 size, SV **strp)

template <class type>
static bool matrix_to_pdl(SV* rsv, SGMatrix<type> sg_matrix, int typecode)
{
  pdl* it = PDL->pdlnew();
  if(!it) {
    return false;
  }
  STRLEN clen = sizeof(type)
    * size_t(sg_matrix.num_rows)
    * size_t(sg_matrix.num_cols);
  PDL_Long dims[2] = {sg_matrix.num_cols, sg_matrix.num_rows};
  PDL->setdims(it, dims, 2);
  it->datatype = typecode;
  PDL->allocdata(it);
  void* data = PDL_REPRP(it);
  if(!data) {
    PDL->destroy(it);
    return false;
  }
  memcpy((type*) data, sg_matrix.matrix, clen);
  PDL->SetSV_PDL(rsv, it);
  return true;
}


  /*
  make piddle belonging to 'class' and of type 'type'
 from avref 'array_ref' which is checked for being
 rectangular first
  */
#if 0
    int is_new_object;
    SV* array = make_contiguous(obj, &is_new_object, -1, typecode, true);
    if (!array)
        return false;
    int32_t ndim = PyArray_NDIM(array);
    if (ndim <= 0)
      return false;
    int32_t* temp_dims = SG_MALLOC(int32_t, ndim);
    npy_intp* py_dims = PyArray_DIMS(array);
    for (int32_t i=0; i<ndim; i++)
      temp_dims[i] = py_dims[i];
    sg_array = SGNDArray<type>((type*) PyArray_BYTES(array), temp_dims, ndim);
    ((AV*) array)->flags &= (-1 ^ NPY_OWNDATA);
    Py_DECREF(array);
#endif



template <class type>
static bool array_from_pdl(SGNDArray<type>& sg_array, SV* obj, int typecode)
{
    int level = 0;
    
    pdl* it = PDL->SvPDLV(obj);
    PDL->make_physdims(it);

    int32_t ndims_pdl = it->ndims; //could try pdl::getndims(it)
    int32_t* dims_sg = SG_MALLOC(int32_t, ndims_pdl);

    PDL_Long* dims_pdl = it->dims; 
    PDL_Long* inds = (PDL_Long*) pdl_malloc(sizeof(PDL_Long) * it->ndims); /* GCC -> on stack :( */
    void* data_pdl = PDL_REPRP(it);
    PDL_Long* incs_pdl = (PDL_VAFFOK(it) ? it->vafftrans->incs : it->dimincs);
    PDL_Long offs_pdl = PDL_REPROFFS(it);
    double pdl_val, pdl_badval;
    type* data_p = (type*) data_pdl;
    int lind = 0;
    int stop = 0;
    int badflag = (it->state & PDL_BADVAL) > 0;
    if(badflag) {
      pdl_badval = pdl_get_pdl_badvalue(it);
    }


    for(int i = 0; i < ndims_pdl; i++) {
      //int jl = ndims_pdl - j + level;	  
      *(dims_sg + i) = dims_pdl[i]; //also pdl::getdim(it, j)
      inds[i] = 0;
    }

    PDL->make_physdims(it);
    PDL_Long nvals_pdl = it->nvals;
    
    //from set_datatype
    PDL->make_physical(it); /* Wasteful because linearise the array*/
    if(it->trans) {
      pdl_destroytransform(it->trans, 1);
    }
    /*     if(! (a->state && PDL_NOMYDIMS)) { */
    pdl_converttype(&it, typecode, PDL_PERM); //_TMP?

    //pdl_make_physvaffine(it);    
    while(!stop) {
      pdl_val = pdl_at(data_pdl, it->datatype, inds, dims_pdl, incs_pdl, offs_pdl, ndims_pdl);
      //if(badflag && pdl_val == pdl_badval) {
      //sv = newSVpvn( "BAD", 3 ); 
      //change it to a NON_ variable
      //}
      *(data_p + lind) = (type) pdl_val;//pdl_val is double
      lind++;
      stop = 1;
      for(int i = 0; i < it->ndims; i++) {
	if(++(inds[i]) >= it->dims[i]) {
	  inds[i] = 0;
	} else {
	  stop = 0;
	  break;
	}
      }
    }    
    sg_array = SGNDArray<type>(data_p, dims_sg, ndims_pdl);
    //PTZ120929 referencing...?
    return true;
}




//(pdl...)array of strings...to SGDtringList
#if 0
      char*     sv_2pv_nolen(SV* sv);
      bool    is_utf8_string(const U8 *s, STRLEN len);
      STRLEN  is_utf8_char(const U8 *s);
      //"PUTBACK" before and "SPAGAIN"STRLEN  PL_na SV_GMAGIC
      //typecode is flags SVf_UTF8, or SVs_TEMP
      I32 el_len = unpackstring(const char *pat
				, const char *patend
				, const char *s
				, const char *strend
				, U32 flags);
      SV*     newSVpvs_flags(const char* s, U32 flags);
      char*   sv_2pv_flags(SV *const sv, STRLEN *const lp, const I32 flags);
      //strlen((SV*) *el_psv);
      const char* el_str = SvPVutf8_nolen(*el_psv);
      el_len = is_utf8_char(el_str);
      if(!el_len) {
	el_len = is_utf8_char(el_str);
      }
#endif

#if 0
/* Check if is a list */
if (!list || PyList_Check(list) || PyList_Size(list)==0)
  {
    int32_t size=PyList_Size(list);
    shogun::SGString<type>* strings=SG_MALLOC(shogun::SGString<type>, size);

    int32_t max_len=0;
    for (int32_t i=0; i<size; i++)
      {
	new (&strings[i]) SGString<type>();
	SV *o = PyList_GetItem(list,i);
	if (typecode == NPY_STRING || typecode == NPY_UNICODE)
	  {
	    if (PyUnicode_Check(o))
	      if (PyString_Check(o))
                {

		  int32_t len = PyUnicode_GetSize((SV*) o);
		  const char* str = PyBytes_AsString(PyUnicode_AsASCIIString(const_cast<SV*>(o)));
		  int32_t len = PyString_Size(o);
		  const char* str = PyString_AsString(o);
		  max_len=shogun::CMath::max(len,max_len);
		  strings[i].slen=len;
		  strings[i].string=NULL;

		  if (len>0)
                    {
		      strings[i].string=SG_MALLOC(type, len);
		      memcpy(strings[i].string, str, len);
                    }
                }
	      else
                {
		  PyErr_SetString(PyExc_TypeError, "all elements in list must be strings");

		  for (int32_t j=0; j<i; j++)
		    strings[i].~SGString<type>();
		  SG_FREE(strings);
		  return false;
                }
	  }
	else
	  {
	    if (is_array(o) && array_dimensions(o)==1 && array_type(o) == typecode)
	      {
		int is_new_object=0;
		SV* array = make_contiguous(o, &is_new_object, 1, typecode);
		if (!array)
		  return false;

		type* str=(type*) PyArray_BYTES(array);
		int32_t len = PyArray_DIM(array,0);
		max_len=shogun::CMath::max(len,max_len);

		strings[i].slen=len;
		strings[i].string=NULL;

		if (len>0)
		  {
		    strings[i].string=SG_MALLOC(type, len);
		    memcpy(strings[i].string, str, len*sizeof(type));
		  }

		if (is_new_object)
		  Py_DECREF(array);
	      }
	    else
	      {
		PyErr_SetString(PyExc_TypeError, "all elements in list must be of same array type");

		for (int32_t j=0; j<i; j++)
		  strings[i].~SGString<type>();
		SG_FREE(strings);
		return false;
	      }
	  }
      }

        SGStringList<type> sl;
        sl.strings=strings;
        sl.num_strings=size;
        sl.max_string_length=max_len;
        sg_strings=sl;

        return true;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a/empty list");
        return false;
    }

    shogun::SGString<type>* l_ss;
    int l_sz;

    int32_t max_len = 0;

    PDL_Long * inds;
    PDL_Long * incs;
    PDL_Long offs;
    void *data;
    int ind;
    int lind;
    int stop = 0;
    AV *av;
    
    SV *sv;
    SV** pdl_val;
    SV** data_psv = (SV**) data;

    //nelem
    PDL->make_physdims(i_pdl);

    //list_c
    //PDL->make_physavaffine(i_pdl);

    //SV* pdl_core_listref_c(pdl* x)
    int inds = (PDL_Long*) pdl_malloc(sizeof(PDL_Long) * i_pdl->ndims); /* GCC -> on stack :( */
    void* data = PDL_REPRP(i_pdl);

    
    incs = (PDL_VAFFOK(i_pdl) ? i_pdl->vafftrans->incs : i_pdl->dimincs);
    offs = PDL_REPROFFS(i_pdl);
    l_sz = i_pdl->nvals;
    l_ss = SG_MALLOC(shogun::SGString<type>, l_sz);

    lind = 0;
    for(int ind=0; ind < i_pdl->ndims; ind++) inds[ind] = 0;
    while(!stop && lind < l_sz) {
      STRLEN el_len;
      const char* el_str;
      SV* el_pv;
      //pdl_val = cast<SV**>(pdl_at(data, x->datatype, inds, x->dims, incs, offs, x->ndims));
      int i = pdl_get_offset(inds, i_pdl->dims, incs, offs, i_pdl->ndims);
      //cast
      //pdl_val shall be SVpv references...
      
      if(SvROK(data_psv[i]) && (SvTYPE(SvRV(data_psv[i])) == SVt_PVMG)) {
	el_pv = SvRV((SV*) data_psv[i]);
	el_str = SvPV(el_pv, el_len);
      } else {
	el_len = 0;
      }
      //el_psv = PTR2RV(pdl_val);
      new(&l_ss[lind]) SGString<type>();
      //PTZ121002 free *el_psv?
      l_ss[i].slen = el_len;
      l_ss[i].string = NULL;
      if (el_len > 0) {
	l_ss[i].string = SG_MALLOC(type, el_len);
	memcpy(l_ss[i].string, el_str, el_len);
      }
      lind++;
      stop = 1;
      for(ind = 0; ind < i_pdl->ndims; ind++) {
	if(++(inds[ind]) >= i_pdl->dims[ind]) {
	  inds[ind] = 0;
	} else {
	  stop = 0; break;
	}
      }
    }

#endif
#if 0
      if(SvROK(datasv[i]) && (SvTYPE(SvRV(data_psv[i])) == SVt_PVMG)) {
	//pdl_val shall be SVpv references...
	el_pv = SvRV((SV*) data_psv[i]);
	el_str = SvPV(el_pv, el_len);
      }
      //el_psv = PTR2RV(pdl_val);
      //PTZ121002 free *el_psv?
      //pdl_val = cast<SV**>(pdl_at(data, x->datatype, inds, x->dims, incs, offs, x->ndims));


    //list_c
    //pdl_make_physavaffine(i_pdl);

    //SV* pdl_core_listref_c(pdl* x)

    //double pdl_get(pdl *it,int *inds);//quite inefficient!

    //PTZ121011 I can see it is not good!!
    // datasv ->data , array of pv*, values ???
    // how does it translates with avaffine?
    //do I need to sacrify dims[-1] and stuff it into the StringList?
    //List is actually a vector...
    //nelem
    //PDL->make_physdims(i_pdl);

#endif
#if 0
      SV* datasv = (SV*) it->datasv;
      if(datasv) {
	if(SvOK(datasv) && (SvTYPE(datasv) == SVt_PVMG)) {
	  el_pv =  datasv;
	  el_str = SvPV(el_pv, el_len);
	} else
	if(SvROK(datasv) && (SvTYPE(SvRV(datasv)) == SVt_PVMG)) {
	  el_pv = SvRV((SV*) datasv);
	  el_str = SvPV(el_pv, el_len);	
	}
      }

      //PTZ121011 i have got he feeling a dim needs to be sacrified
      // most likely the last one holds max(len) of strings.
      //PTZ121011 might have to transpose this algo

#endif

/*
 * typically, a PDL::Char dimension will be.... 
 * ...(strlen~columns)_0 x rows_1 x .... x (size_t)_last
 *
 * we are looking for a sequence of strings.
 *
 * 
 */

template <class type>
static bool string_from_pdl(SGStringList<type>& sg_strings, SV* sv, U32 typecode)
{
  pdl* it = if_piddle(sv);
  if(it) {
    it = pdl_get_convertedpdl(it, typecode);
    PDL->make_physical(it); /* Wasteful*/  
    if(!PDL_ENSURE_ALLOCATED(it)) {
      pdl_dump(it);
      return false;
    }
    void* data = PDL_REPRP(it);    
    PDL_Long* incs = (PDL_VAFFOK(it) ? it->vafftrans->incs : it->dimincs);
    PDL_Long  offs = PDL_REPROFFS(it);
    int ndims = it->ndims - 1;
    PDL_Long l_len_max;
    if(it->ndims > 2) {
      l_len_max = it->dims[ndims] * it->dims[0];
    } else if(it->ndims == 2) {
      l_len_max = it->dims[ndims];     
      ndims = it->ndims;
    } else {
      warn("not enough dimensions");
     return false;
    }
    if(!l_len_max) {
      warn("null dimension value");
      return false;
    }
    int l_sz = it->nvals / l_len_max;
    shogun::SGString<type>* l_ss = SG_MALLOC(shogun::SGString<type>, l_sz);
    if(!l_ss) {
      return false;
    }
    sg_strings.strings = l_ss;
    sg_strings.num_strings = l_sz;
    sg_strings.max_string_length = 0;

    PDL_Long* inds = (PDL_Long*) pdl_malloc(sizeof(PDL_Long) * it->ndims);
    if(!inds) {
      SG_FREE(l_ss);
      return false;
    }
    for(int i = 0; i < it->ndims; i++) inds[i] = 0;
    int lind = 0;
    int stop = 0;
    while(!stop && lind < l_sz) {
      STRLEN el_len = 0;
      const char* el_str;
      SV* el_pv;
      int i = pdl_get_offset(inds, it->dims, incs, offs, it->ndims);
      if(i >= l_sz) {
	warn("offset error in string conversion::bayling out");
	for(int32_t j = 0; j < lind; j++)
	  l_ss[j].~SGString<type>();
	SG_FREE(l_ss);
	//free(inds);
	return false;
      }
      //PTZ121012 not sure about number types.
      el_str = (char*) data + i;
      el_len = strnlen(el_str, l_len_max);
      new(&l_ss[lind]) SGString<type>();
      l_ss[lind].slen = el_len;
      l_ss[lind].string = NULL;
      if(el_len > 0) {
	l_ss[lind].string = SG_MALLOC(type, el_len);
	if(!l_ss[lind].string) {
	  for(int32_t j = 0; j <= lind; j++)
	    l_ss[j].~SGString<type>();
	  SG_FREE(l_ss);
	  //free(inds);
	  return false;	  
	}
	memcpy(l_ss[lind].string, el_str, el_len);
	if(el_len > sg_strings.max_string_length) {
	  sg_strings.max_string_length = el_len;
	}
      }
      lind++;
      stop = 1;
      for(int n = 0; n < ndims; n++) {
	if(++(inds[n]) >= it->dims[n]) {
	  inds[n] = 0;
	} else {
	  stop = 0;
	  break;
	}
      }
    }
    //free(inds);
    return true;
  }
  return false;
}


template <class type>
static bool string_from_perl(SGStringList<type>& sg_strings, SV* sv, U32 typecode)
{
  //PTZ121011 pure perl, but not used create another typemap...like string_from_perl
  sg_strings->max_string_length = 0;
  if(is_array(sv)) {
    AV* l_av = (AV*) SvRV(sv);   /* dereference */
    int l_sz = av_len(l_av) + 1;
    shogun::SGString<type>* l_ss = SG_MALLOC(shogun::SGString<type>, l_sz);
    sg_strings.strings = l_ss;
    for (int32_t i = 0; i <= l_sz; i++) {
      STRLEN el_len;
      SV** el_psv = av_fetch(l_av, i, 0);
      const char* el_str = SvPV(*el_psv, el_len);
      new (&l_ss[i]) SGString<type>();
      //PTZ121002 free *el_psv?
      l_ss[i].slen = el_len;
      l_ss[i].string = NULL;
      if (el_len > 0) {
	l_ss[i].string = SG_MALLOC(type, el_len);
	memcpy(l_ss[i].string, el_str, el_len);
	if(el_len > sg_strings.max_string_length) {
	  sg_strings.max_string_length = el_len;
	}
      }
    }
    sg_strings.num_strings = l_sz;
    return true;
  }
  return false;
}

#if 0
SV* list = PyList_New(num);
if (list && str)
  {
    for (int32_t i=0; i<num; i++)
      {
	SV* s=NULL;

	if (typecode == NPY_STRING || typecode == NPY_UNICODE)
	  {
	    /* This path is only taking if str[i].string is a char*. However this cast is
	       required to build through for non char types. */
	    s=PyUnicode_FromStringAndSize((char*) str[i].string, str[i].slen);
	    s=PyString_FromStringAndSize((char*) str[i].string, str[i].slen);
	  }
	else
	  {
	    PyArray_Descr* descr=PyArray_DescrFromType(typecode);
	    type* data = SG_MALLOC(type, str[i].slen);
	    if (descr && data)
	      {
		memcpy(data, str[i].string, str[i].slen*sizeof(type));
		npy_intp dims = str[i].slen;

		s = PyArray_NewFromDescr(&PyArray_Type,
					 descr, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
		((AV*) s)->flags |= NPY_OWNDATA;
	      }
	    else
	      return false;
	  }

	PyList_SetItem(list, i, s);
      }
    obj = list;
    return true;
  }
 else
   return false;
#endif


#if 0
//each string is a ref to a pv
//ST(0) ...is given to first argument
  //from void pdl_makescratchhash(pdl *ret,double data, int datatype)
  //(SV*) -> PDL_Long ??pdl_whichdatatype((SV*));
SV** xx_pdl;
  //it->data = pdl_malloc(pdl_howbig(it->datatype) * num);
  //double packing in pdl?
  //SV* dat_sv = newSVpv((char*)it->data, pdl_howbig(it->datatype) * num);
  //it->data = SvPV(dat_sv, el_len);
  //it->datasv = dat_sv;
  //sv_2mortal(getref_pdl(it));
  //PTZ121004SvPDLV(*psv); might have done the job here , and so pre free it->data,
  // free it->data it->datasv ??
  //SetSV_PDL(*psv, it_pdl);
  //dims_pdl[0] = num;
  //PDL->setdims(it, dims_pdl, 1);
  //pdl_makescratchhash(pdl *ret,double data, int datatype)
  //pdl_set_type();
  //xx_pdl = (SV**) data_pdl;
  //pdl_make_physvaffine( it );
  //PDL_Long* inds_pdl = (PDL_Long*) pdl_malloc( sizeof( PDL_Long ) * it->ndims);
    //SV* s_sv;
    //void    sv_setpvn_mg(SV *const sv, const char *const ptr, const STRLEN len)
    //sv_setpvn_mg(s_sv, (char*) str[i].string, str[i].slen);
    //store into pdl    //pdl_set(i);
    //int pdl_i = pdl_get_offset(inds_pdl, it->dims, incs_pdl, offs_pdl, it->ndims);
    //xx_pdl[i] = SvRV(s_sv);

#endif

#if 0
    stop = 1;
    for(int ind_pdl = 0; ind_pdl < ndims_pdl - 1; ind_pdl++) {
      if(++(inds_pdl[ind_pdl]) >= it->dims[ind_pdl]) {
	inds_pdl[ind_pdl] = 0;
      } else {
	stop = 0; break;
      }
    }

  PDL_Long* incs_pdl = (PDL_VAFFOK(it) ? it->vafftrans->incs : it->dimincs);
  int offs_pdl = PDL_REPROFFS(it);
  //for(int ind_pdl = 0; ind_pdl < it->ndims; ind_pdl++) inds_pdl[ind_pdl] = 0;
  //int lind_pdl = 0;
  //int stop = 0;
  //for(int n = 0; n < sg_num; n++) {
  //  if(sg_str[n].slen > sg_slen_max) sg_slen_max = sg_str[n].slen;
  //}

#endif



template <class type>
static bool string_to_pdl(SV* rsv, SGStringList<type> sg_strings, int typecode)
{
  pdl* it = PDL->pdlnew();
#if 0

  //bless it to a PDL::Char...
  char objname[] = "PDL::Char";
  //HV *bless_stash = 0;
  SV *y_SV;

  PUSHMARK(SP);
  XPUSHs(sv_2mortal(newSVpv(objname, 0)));
  PUTBACK;
  perl_call_method("initialize", G_SCALAR);
  SPAGAIN;
  rsv = POPs;
  PUTBACK;
  it = PDL->SvPDLV(rsv);
  //int ndims_pdl = 2;

#endif
  if(!it) {
    return false;
  }
  shogun::SGString<type>* sg_str = sg_strings.strings;
  int32_t sg_num = sg_strings.num_strings;
  //work out max len!
  STRLEN sg_slen_max = sg_strings.max_string_length;
  if(sg_slen_max <= 0) {
    warn("this is an all-null string dimension");
    return false;
  }
  PDL_Long dims_pdl[3] = {sg_slen_max, sg_num, 1};
  PDL->setdims(it, dims_pdl, 3);
  it->datatype = typecode;
  PDL->allocdata(it);
  void* data_pdl = PDL_REPRP(it);
  if(!data_pdl) {
    PDL->destroy(it);
    return false;
  }
  for(int32_t i = 0; i < sg_num; i++) {
    //PTZ121012 really to check this with unicode types also...  
    memcpy((type*) data_pdl + (i * sg_slen_max), sg_str[i].string, sizeof(type) * sg_str[i].slen);
    //PTZ121012 shall have calloced also...
  }
  PDL->SetSV_PDL(rsv, it);
  return true;
}




#if 0
    if(!is_pdl_sparse_matrix(o, typecode))
    {
      warn("not a column compressed sparse matrix");
      return false;
    }
#endif
    /* get data array */
    //int is_new_object_data = 0;
    //SV* array_data_sv = make_contiguous(data, &is_new_object_data, 1, typecode);
    //type* bytes_data=(type*) PyArray_BYTES(array_data);
    //int32_t len_data = PyArray_DIM(array_data,0);
    //PTZ120928 expect "indptr","indices", "data","shape" in HASH of PDL?!!
    //PTZ120928 so far I put it like this: data, idx(ices), ind(ptr)!!!


template <class type>
static bool spmatrix_from_pdl(SGSparseMatrix<type>& sg_matrix, SV* obj, int typecode)
{
    if(!(SvROK(obj) && SvTYPE(SvRV(obj)) == SVt_PVAV))
      return false;

    AV* tuple_av = (AV*) SvRV(obj);

    //HV* hash = (HV*) SvRV( obj );
    //PDL_Long *dims;
    //*ndims = (int) av_len(array) + 1;
    AV* data_av  = (AV*) SvRV(*(av_fetch(tuple_av, 0, 0)));
    AV* idx_av   = (AV*) SvRV(*(av_fetch(tuple_av, 1, 0)));
 
    PDL_Long len_indices = (int) av_len(idx_av) + 1;
    PDL_Long len_data = (int) av_len(data_av) + 1;

    if(len_indices != len_data) {
      //PTZ120928free data_av,idx_av!!!
      return false;
    }
    AV* ind_av   = (AV*) SvRV(*(av_fetch(tuple_av, 2, 0)));
    PDL_Long len_indptr = (int) av_len(ind_av) + 1;

    AV* shape_av = (AV*) SvRV(*(av_fetch(tuple_av, 3, 0)));
    AV* shape_feat_av = (AV*) SvRV(*(av_fetch(shape_av, 0, 0)));
    AV* shape_vec_av = (AV*) SvRV(*(av_fetch(shape_av, 1, 0)));
    PDL_Long num_feat = (int) av_len(shape_feat_av) + 1;
    PDL_Long num_vec = (int) av_len(shape_vec_av) + 1;
    shogun::SGSparseVector<type>* sfm = SG_MALLOC(shogun::SGSparseVector<type>, num_vec);

    for(int32_t i = 0; i < num_vec; i++) {
      new (&sfm[i]) SGSparseVector<type>();
    }
    int32_t num_i_prev = 0;
    int32_t ij = 0;
    for(int32_t i = 1; i < len_indptr; i++) {
      //SV* ind_i_iv = *(av_fetch(ind_av, i, 0));
      int32_t num_i = SvIV( *(av_fetch(ind_av, i, 0)) );
      int32_t num = num_i - num_i_prev;
      num_i_prev = num_i;
      if(num > 0) {
	shogun::SGSparseVectorEntry<type>* features = SG_MALLOC(shogun::SGSparseVectorEntry<type>, num);
	for (int32_t j = 0; j < num; j++) {
	  features[j].feat_index = SvIV( *(av_fetch(idx_av, ij, 0)) );
	  features[j].entry      = SvNV( *(av_fetch(data_av, ij, 0)) );
	  ij++;
	}
	sfm[i - 1].num_feat_entries = num;
	sfm[i - 1].features = features;
      }
    }
    //dereference shape_av, tuple_av, data_av, idx_av.../?
    SGSparseMatrix<type> sm;
    sm.sparse_matrix = sfm;
    sm.num_features = num_feat;
    sm.num_vectors = num_vec;
    sg_matrix = sm;
    return true;
}


#if 0

    /* fetch sparse attributes */
    SV* indptr = SV_GetAttrString(o, "indptr");
    SV* indices = SV_GetAttrString(o, "indices");
    SV* data = SV_GetAttrString(o, "data");
    SV* shape = SV_GetAttrString(o, "shape");

    /* check that types are OK */
    if ((!is_array(indptr)) || (array_dimensions(indptr)!=1) ||
            (array_type(indptr)!=NPY_INT && array_type(indptr)!=NPY_LONG))
    {
        PyErr_SetString(PyExc_TypeError,"indptr array should be 1d int's");
        return false;
    }

    if (!is_array(indices) || array_dimensions(indices)!=1 ||
            (array_type(indices)!=NPY_INT && array_type(indices)!=NPY_LONG))
    {
        PyErr_SetString(PyExc_TypeError,"indices array should be 1d int's");
        return false;
    }

    if (!is_array(data) || array_dimensions(data)!=1 || array_type(data) != typecode)
    {
        PyErr_SetString(PyExc_TypeError,"data array should be 1d and match datatype");
        return false;
    }

    if (!PyTuple_Check(shape))
    {
        PyErr_SetString(PyExc_TypeError,"shape should be a tuple");
        return false;
    }

    /* get array dimensions */
    int32_t num_feat=PyLong_AsLong(PyTuple_GetItem(shape, 0));
    int32_t num_vec=PyLong_AsLong(PyTuple_GetItem(shape, 1));
    int32_t num_feat=PyInt_AsLong(PyTuple_GetItem(shape, 0));
    int32_t num_vec=PyInt_AsLong(PyTuple_GetItem(shape, 1));

    /* get indptr array */
    int is_new_object_indptr=0;
    SV* array_indptr = make_contiguous(indptr, &is_new_object_indptr, 1, NPY_INT32);
    if (!array_indptr) return false;
    int32_t* bytes_indptr=(int32_t*) PyArray_BYTES(array_indptr);
    int32_t len_indptr = PyArray_DIM(array_indptr,0);

    /* get indices array */
    int is_new_object_indices=0;
    SV* array_indices = make_contiguous(indices, &is_new_object_indices, 1, NPY_INT32);
    if (!array_indices) return false;
    int32_t* bytes_indices=(int32_t*) PyArray_BYTES(array_indices);
    int32_t len_indices = PyArray_DIM(array_indices,0);

    /* get data array */
    int is_new_object_data=0;
    SV* array_data = make_contiguous(data, &is_new_object_data, 1, typecode);
    if (!array_data) return false;
    type* bytes_data=(type*) PyArray_BYTES(array_data);
    int32_t len_data = PyArray_DIM(array_data,0);

    if (len_indices!=len_data)
        return false;

    shogun::SGSparseVector<type>* sfm = SG_MALLOC(shogun::SGSparseVector<type>, num_vec);

    for (int32_t i=0; i<num_vec; i++)
        new (&sfm[i]) SGSparseVector<type>();

    for (int32_t i=1; i<len_indptr; i++)
    {
        int32_t num = bytes_indptr[i]-bytes_indptr[i-1];
        
        if (num>0)
        {
            shogun::SGSparseVectorEntry<type>* features=SG_MALLOC(shogun::SGSparseVectorEntry<type>, num);

            for (int32_t j=0; j<num; j++)
            {
                features[j].feat_index=*bytes_indices;
                features[j].entry=*bytes_data;

                bytes_indices++;
                bytes_data++;
            }
            sfm[i-1].num_feat_entries=num;
            sfm[i-1].features=features;
        }
    }

    if (is_new_object_indptr)
        Py_DECREF(array_indptr);
    if (is_new_object_indices)
        Py_DECREF(array_indices);
    if (is_new_object_data)
        Py_DECREF(array_data);

    Py_DECREF(indptr);
    Py_DECREF(indices);
    Py_DECREF(data);
    Py_DECREF(shape);

    SGSparseMatrix<type> sm;
    sm.sparse_matrix=sfm;
    sm.num_features=num_feat;
    sm.num_vectors=num_vec;
    sg_matrix=sm;
#endif






template <class type>
static bool spmatrix_to_pdl(SV* rsv, SGSparseMatrix<type> sg_matrix, int typecode)
{
  shogun::SGSparseVector<type>* sfm = sg_matrix.sparse_matrix;
    AV* tuple_av = newAV();
    AV* ind_av = newAV();
    int32_t ind_i = 0;
    av_store(ind_av, 0, newSViv((IV) ind_i));

    AV* data_av = newAV();
    AV* idx_av = newAV();
    int32_t ij = 0;
    //PTZ120928 so slow....
    for(int32_t i = 0; i < sg_matrix.num_vectors; i++) {
      ind_i += sfm[i].num_feat_entries;
      av_store(ind_av, i + 1, newSViv((IV) ind_i));
      for(int32_t j = 0; j < sfm[i].num_feat_entries; j++) {
	av_store(idx_av,  ij, newSViv((IV) sfm[i].features[j].feat_index));
	//PTZ120928 here typecode shall be used...
	av_store(data_av, ij, newSVuv((NV) sfm[i].features[j].entry     ));
	ij++;
      }
    }
    //PTZ120928 could do with Tie magic..might need to reference AV*???SvRV(_av)
    //PTZ121004 and add it into a pdl!
    av_store(tuple_av, 0, (SV*)data_av);
    av_store(tuple_av, 1, (SV*)idx_av );

    //SV*     newSVrv(SV *const rv, const char *const classname)
    //SV* sv_pdl = newSVrv(rsv, "PDL");
    sv_setsv_mg(rsv, *av_store(tuple_av, 2, (SV*)ind_av));

    return true;
 fail:
    return false;
}

#if 0
    SV* tuple = PyTuple_New(3);
    if (tuple && sfm)
    {
        SV* data_py=NULL;
        SV* indices_py=NULL;
        SV* indptr_py=NULL;

        PyArray_Descr* descr=PyArray_DescrFromType(PDL_INT32);
        PyArray_Descr* descr_data=PyArray_DescrFromType(typecode);

        int32_t* indptr = SG_MALLOC(int32_t, num_vec+1);
        int32_t* indices = SG_MALLOC(int32_t, nnz);
        type* data = SG_MALLOC(type, nnz);

        if (descr && descr_data && indptr && indices && data)
        {
            indptr[0]=0;

            int32_t* i_ptr=indices;
            type* d_ptr=data;

            for (int32_t i=0; i<num_vec; i++)
            {
                indptr[i+1]=indptr[i];
                indptr[i+1]+=sfm[i].num_feat_entries;

                for (int32_t j=0; j<sfm[i].num_feat_entries; j++)
                {
                    *i_ptr=sfm[i].features[j].feat_index;
                    *d_ptr=sfm[i].features[j].entry;

                    i_ptr++;
                    d_ptr++;
                }
            }

            npy_intp indptr_dims = num_vec+1;
            indptr_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &indptr_dims, NULL, (void*) indptr, PDL_FARRAY | PDL_WRITEABLE, NULL);
            ((AV*) indptr_py)->flags |= PDL_OWNDATA;

            npy_intp dims = nnz;
            indices_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &dims, NULL, (void*) indices, PDL_FARRAY | PDL_WRITEABLE, NULL);
            ((AV*) indices_py)->flags |= PDL_OWNDATA;

            data_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr_data, 1, &dims, NULL, (void*) data, PDL_FARRAY | PDL_WRITEABLE, NULL);
            ((AV*) data_py)->flags |= PDL_OWNDATA;

            PyTuple_SetItem(tuple, 0, data_py);
            PyTuple_SetItem(tuple, 1, indices_py);
            PyTuple_SetItem(tuple, 2, indptr_py);
            obj = tuple;
            return true;
        }
        else
            return false;
    }
    else
        return false;
#endif

#if 0
    SV* tuple = PyTuple_New(2);
npy_intp dims = sg_vector.num_feat_entries;
    if (!tuple) return false;
    SV* data_py=NULL;
    SV* indices_py=NULL;
    PyArray_Descr* descr=PyArray_DescrFromType(PDL_INT32);
    PyArray_Descr* descr_data=PyArray_DescrFromType(typecode);
    int32_t* indices = SG_MALLOC(int32_t, dims);
    type* data = SG_MALLOC(type, dims);
    if (!(descr && descr_data && indices && data)) return false;
    int32_t* i_ptr=indices;
    type* d_ptr=data;
    for (int32_t j=0; j<sg_vector.num_feat_entries; j++) {
        *i_ptr=sg_vector.features[j].feat_index;
        *d_ptr=sg_vector.features[j].entry;
        i_ptr++;
        d_ptr++;
    }
    indices_py = PyArray_NewFromDescr(&PyArray_Type,
            descr, 1, &dims, NULL, (void*) indices, PDL_FARRAY | PDL_WRITEABLE, NULL);
    ((AV*) indices_py)->flags |= PDL_OWNDATA;

    data_py = PyArray_NewFromDescr(&PyArray_Type,
            descr_data, 1, &dims, NULL, (void*) data, PDL_FARRAY | PDL_WRITEABLE, NULL);
    ((AV*) data_py)->flags |= PDL_OWNDATA;
    PyTuple_SetItem(tuple, 0, data_py);
    PyTuple_SetItem(tuple, 1, indices_py);
    obj = tuple;
    //PDL_Long features = sg_vector.num_feat_entries;
    //AV* dims_av = newAV();
    //AV* data_i_av;
    //int32_t* indices = SG_MALLOC(int32_t, features);
    //int32_t* idx_p = indices;
    //type* data = SG_MALLOC(type, features);
    //type* data_p = data;
#endif


template <class type>
static bool spvector_to_pdl(SV* rsv, SGSparseVector<type> sg_vector, int typecode)
{
    AV* tuple_av = newAV();
    AV* data_av = newAV();
    AV* idx_av = newAV();

    for (int32_t j = 0; j < sg_vector.num_feat_entries; j++) {
      av_store(idx_av,  j, newSViv((IV) sg_vector.features[j].feat_index));
      av_store(data_av, j, newSVuv((NV) sg_vector.features[j].entry     ));

      //PTZ120929 check swig ways also...for handling values
    }
    //PTZ120928 could do with Tie magic..
    av_store(tuple_av, 0, newRV((SV*)data_av));
    //*obj = *av_store(tuple_av, 1, newRV((SV*)idx_av));
    sv_setsv_mg(rsv, *av_store(tuple_av, 1, newRV((SV*)idx_av)));
    return true;
}


#if 0
  /* Install constant */
  for (i = 0; swig_constants[i].type; i++) {
    SV *sv;
    sv = get_sv((char*)swig_constants[i].name, TRUE | 0x2 | GV_ADDMULTI);
    switch(swig_constants[i].type) {
    case SWIG_INT:
      sv_setiv(sv, (IV) swig_constants[i].lvalue);
      break;
    case SWIG_FLOAT:
      sv_setnv(sv, (double) swig_constants[i].dvalue);
      break;
    case SWIG_STRING:
      sv_setpv(sv, (char *) swig_constants[i].pvalue);
      break;
    case SWIG_POINTER:
      SWIG_MakePtr(sv, swig_constants[i].pvalue, *(swig_constants[i].ptype),0);
      break;
    case SWIG_BINARY:
      SWIG_MakePackedObj(sv, swig_constants[i].pvalue, swig_constants[i].lvalue, *(swig_constants[i].ptype));
      break;
    default:
      break;
    }
    SvREADONLY_on(sv);
  }
 #endif
%}

/* CFeatures to ... */
%define FEATURES_BY_TYPECODE(obj, f, type, typecode)
     switch (typecode) {
     case F_BOOL:
       obj=SWIG_NewPointerObj(f, $descriptor(type<bool> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_CHAR:
       obj=SWIG_NewPointerObj(f, $descriptor(type<char> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_BYTE:
       obj=SWIG_NewPointerObj(f, $descriptor(type<uint8_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_SHORT:
       obj=SWIG_NewPointerObj(f, $descriptor(type<int16_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_WORD:
       obj=SWIG_NewPointerObj(f, $descriptor(type<uint16_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_INT:
       obj=SWIG_NewPointerObj(f, $descriptor(type<int32_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_UINT:
       obj=SWIG_NewPointerObj(f, $descriptor(type<uint32_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_LONG:
       obj=SWIG_NewPointerObj(f, $descriptor(type<int64_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_ULONG:
       obj=SWIG_NewPointerObj(f, $descriptor(type<uint64_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_SHORTREAL:
       obj=SWIG_NewPointerObj(f, $descriptor(type<float32_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_DREAL:
       obj=SWIG_NewPointerObj(f, $descriptor(type<float64_t> *), SWIG_POINTER_EXCEPTION);
       break;
     case F_LONGREAL:
       obj=SWIG_NewPointerObj(f, $descriptor(type<floatmax_t> *), SWIG_POINTER_EXCEPTION);
       break;
     default:
       obj=SWIG_NewPointerObj(f, $descriptor(shogun::CFeatures*), SWIG_POINTER_EXCEPTION);
       break;
     }
%enddef

%typemap(out) shogun::CFeatures*
{
  int feats_class=$1->get_feature_class();
  int feats_type=$1->get_feature_type();
  switch (feats_class){
  case C_DENSE:
    {
      FEATURES_BY_TYPECODE($result, $1, shogun::CDenseFeatures, feats_type)
	break;
    }    
  case C_SPARSE:
    {
      FEATURES_BY_TYPECODE($result, $1, shogun::CSparseFeatures, feats_type)
	break;
    }    
  case C_STRING:
    {
      FEATURES_BY_TYPECODE($result, $1, shogun::CStringFeatures, feats_type)
	break;
    }
  case C_COMBINED:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CCombinedFeatures*), SWIG_POINTER_EXCEPTION);
    break;    
  case C_COMBINED_DOT:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CCombinedDotFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  case C_WD:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CWDFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  case C_SPEC:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CExplicitSpecFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  case C_WEIGHTEDSPEC:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CImplicitWeightedSpecFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  case C_POLY:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CPolyFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  case C_STREAMING_DENSE:
    {
      FEATURES_BY_TYPECODE($result, $1, shogun::CStreamingDenseFeatures, feats_type)
	break;
    }
  case C_STREAMING_SPARSE:
    {
      FEATURES_BY_TYPECODE($result, $1, shogun::CStreamingSparseFeatures, feats_type)
	break;
    }    
  case C_STREAMING_STRING:
    {
      FEATURES_BY_TYPECODE($result, $1, shogun::CStreamingStringFeatures, feats_type)
	break;
    }
  case C_STREAMING_VW:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CStreamingVwFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  case C_BINNED_DOT:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CBinnedDotFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  case C_DIRECTOR_DOT:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CDirectorDotFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  default:
    $result=SWIG_NewPointerObj($1, $descriptor(shogun::CFeatures*), SWIG_POINTER_EXCEPTION);
    break;
  }
  argvi++;
}

#if 0
//PTZ121011 I believe swigperl has its own?
%typemap(typecheck) const char* 
{
  $1 = is_utf8_char($INPUT);
}

//PyBytes_AsString(PyUnicode_AsASCIIString(const_cast<SV*>($input)));
%typemap(in) const char* 
{
  if(!($1 = newSVpvs_flags($input, 0)))
    SWIG_fail;
}
%typemap(freearg) const char* 
{
  // pass
}
#endif

#define	PDL_BOOL	PDL_B
#define PDL_UNICODE	PDL_US
#define PDL_STRING	PDL_B
#define PDL_UINT8	PDL_B
#define PDL_INT16	PDL_S
#define PDL_UINT16	PDL_US
#define PDL_INT32	PDL_L
#define PDL_UINT32	PDL_L
#define PDL_INT64	PDL_LL
#define PDL_UINT64	PDL_LL
#define PDL_FLOAT32	PDL_F
#define PDL_FLOAT64	PDL_D
#define PDL_LONGDOUBLE	PDL_D
#define PDL_OBJECT	PDL_LL

/* One dimensional input arrays */
%define TYPEMAP_IN_SGVECTOR(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<type>
{
  $1 = is_pdl_vector($input, typecode);
}

%typemap(in) shogun::SGVector<type>
{
  if (!vector_from_pdl<type>($1, $input, typecode))
    SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGVECTOR macros */
TYPEMAP_IN_SGVECTOR(bool,          PDL_BOOL)
TYPEMAP_IN_SGVECTOR(w_char,        PDL_UNICODE)
TYPEMAP_IN_SGVECTOR(char,          PDL_STRING)
TYPEMAP_IN_SGVECTOR(uint8_t,       PDL_UINT8)
TYPEMAP_IN_SGVECTOR(int16_t,       PDL_INT16)
TYPEMAP_IN_SGVECTOR(uint16_t,      PDL_UINT16)
TYPEMAP_IN_SGVECTOR(int32_t,       PDL_INT32)
TYPEMAP_IN_SGVECTOR(uint32_t,      PDL_UINT32)
TYPEMAP_IN_SGVECTOR(int64_t,       PDL_INT64)
TYPEMAP_IN_SGVECTOR(uint64_t,      PDL_UINT64)
TYPEMAP_IN_SGVECTOR(float32_t,     PDL_FLOAT32)
TYPEMAP_IN_SGVECTOR(float64_t,     PDL_FLOAT64)
TYPEMAP_IN_SGVECTOR(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_IN_SGVECTOR(SV*,	   PDL_OBJECT)

#undef TYPEMAP_IN_SGVECTOR

/* One dimensional output arrays */
%define TYPEMAP_OUT_SGVECTOR(type,typecode)
%typemap(out) shogun::SGVector<type>
{
  $result = sv_newmortal();
  if(!vector_to_pdl($result, $1, typecode))
    SWIG_fail;
  if(!is_piddle($result))
    SWIG_fail;
  argvi++;   
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGVECTOR macros */
TYPEMAP_OUT_SGVECTOR(bool,          PDL_BOOL)
TYPEMAP_OUT_SGVECTOR(w_char,        PDL_UNICODE)
TYPEMAP_OUT_SGVECTOR(char,          PDL_STRING)
TYPEMAP_OUT_SGVECTOR(uint8_t,       PDL_UINT8)
TYPEMAP_OUT_SGVECTOR(int16_t,       PDL_INT16)
TYPEMAP_OUT_SGVECTOR(uint16_t,      PDL_UINT16)
TYPEMAP_OUT_SGVECTOR(int32_t,       PDL_INT32)
TYPEMAP_OUT_SGVECTOR(uint32_t,      PDL_UINT32)
TYPEMAP_OUT_SGVECTOR(int64_t,       PDL_INT64)
TYPEMAP_OUT_SGVECTOR(uint64_t,      PDL_UINT64)
TYPEMAP_OUT_SGVECTOR(float32_t,     PDL_FLOAT32)
TYPEMAP_OUT_SGVECTOR(float64_t,     PDL_FLOAT64)
TYPEMAP_OUT_SGVECTOR(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_OUT_SGVECTOR(SV*,	    PDL_OBJECT)

#undef TYPEMAP_OUT_SGVECTOR

/* Two dimensional(rectangular) input arrays */
%define TYPEMAP_IN_SGMATRIX(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGMatrix<type>
{
    $1 = is_pdl_matrix($input, typecode);
}

%typemap(in) shogun::SGMatrix<type>
{
    if(!matrix_from_pdl<type>($1, $input, typecode))
      SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGMATRIX macros */
TYPEMAP_IN_SGMATRIX(bool,          PDL_BOOL)
TYPEMAP_IN_SGMATRIX(w_char,        PDL_UNICODE)
TYPEMAP_IN_SGMATRIX(char,          PDL_STRING)
TYPEMAP_IN_SGMATRIX(uint8_t,       PDL_UINT8)
TYPEMAP_IN_SGMATRIX(int16_t,       PDL_INT16)
TYPEMAP_IN_SGMATRIX(uint16_t,      PDL_UINT16)
TYPEMAP_IN_SGMATRIX(int32_t,       PDL_INT32)
TYPEMAP_IN_SGMATRIX(uint32_t,      PDL_UINT32)
TYPEMAP_IN_SGMATRIX(int64_t,       PDL_INT64)
TYPEMAP_IN_SGMATRIX(uint64_t,      PDL_UINT64)
TYPEMAP_IN_SGMATRIX(float32_t,     PDL_FLOAT32)
TYPEMAP_IN_SGMATRIX(float64_t,     PDL_FLOAT64)
TYPEMAP_IN_SGMATRIX(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_IN_SGMATRIX(SV*,	   PDL_OBJECT)

#undef TYPEMAP_IN_SGMATRIX

/* Two dimensional (rectangular) output arrays */
%define TYPEMAP_OUT_SGMATRIX(type,typecode)
%typemap(out) shogun::SGMatrix<type>
{
  $result = sv_newmortal();
  if(!matrix_to_pdl($result, $1, typecode))
      SWIG_fail;
  if(!is_piddle($result))
    SWIG_fail;
  argvi++;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGMATRIX macros */
TYPEMAP_OUT_SGMATRIX(bool,          PDL_BOOL)
TYPEMAP_OUT_SGMATRIX(w_char,        PDL_UNICODE)
TYPEMAP_OUT_SGMATRIX(char,          PDL_STRING)
TYPEMAP_OUT_SGMATRIX(uint8_t,       PDL_UINT8)
TYPEMAP_OUT_SGMATRIX(int16_t,       PDL_INT16)
TYPEMAP_OUT_SGMATRIX(uint16_t,      PDL_UINT16)
TYPEMAP_OUT_SGMATRIX(int32_t,       PDL_INT32)
TYPEMAP_OUT_SGMATRIX(uint32_t,      PDL_UINT32)
TYPEMAP_OUT_SGMATRIX(int64_t,       PDL_INT64)
TYPEMAP_OUT_SGMATRIX(uint64_t,      PDL_UINT64)
TYPEMAP_OUT_SGMATRIX(float32_t,     PDL_FLOAT32)
TYPEMAP_OUT_SGMATRIX(float64_t,     PDL_FLOAT64)
TYPEMAP_OUT_SGMATRIX(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_OUT_SGMATRIX(SV*,	    PDL_OBJECT)

#undef TYPEMAP_OUT_SGMATRIX

/* N-dimensional input arrays */
%define TYPEMAP_INND(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        shogun::SGNDArray<type>
{
    $1 = is_pdl_array($input, typecode);
}

%typemap(in) shogun::SGNDArray<type>
{
    if(!array_from_pdl<type>($1, $input, typecode))
      SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_INND macros */
TYPEMAP_INND(bool,          PDL_BOOL)
TYPEMAP_INND(w_char,        PDL_UNICODE)
TYPEMAP_INND(char,          PDL_STRING)
TYPEMAP_INND(uint8_t,       PDL_UINT8)
TYPEMAP_INND(int16_t,       PDL_INT16)
TYPEMAP_INND(uint16_t,      PDL_UINT16)
TYPEMAP_INND(int32_t,       PDL_INT32)
TYPEMAP_INND(uint32_t,      PDL_UINT32)
TYPEMAP_INND(int64_t,       PDL_INT64)
TYPEMAP_INND(uint64_t,      PDL_UINT64)
TYPEMAP_INND(float32_t,     PDL_FLOAT32)
TYPEMAP_INND(float64_t,     PDL_FLOAT64)
TYPEMAP_INND(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_INND(SV*,	    PDL_OBJECT)

#undef TYPEMAP_INND

/* input typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_IN(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGStringList<type>
{
    $1 = is_pdl_string($input, typecode);
}

%typemap(in) shogun::SGStringList<type>
{
    if(!string_from_pdl<type>($1, $input, typecode))
        SWIG_fail;
}
%enddef

TYPEMAP_STRINGFEATURES_IN(bool,          PDL_BOOL)
TYPEMAP_STRINGFEATURES_IN(w_char,        PDL_UNICODE)
TYPEMAP_STRINGFEATURES_IN(char,          PDL_STRING)
TYPEMAP_STRINGFEATURES_IN(uint8_t,       PDL_UINT8)
TYPEMAP_STRINGFEATURES_IN(int16_t,       PDL_INT16)
TYPEMAP_STRINGFEATURES_IN(uint16_t,      PDL_UINT16)
TYPEMAP_STRINGFEATURES_IN(int32_t,       PDL_INT32)
TYPEMAP_STRINGFEATURES_IN(uint32_t,      PDL_UINT32)
TYPEMAP_STRINGFEATURES_IN(int64_t,       PDL_INT64)
TYPEMAP_STRINGFEATURES_IN(uint64_t,      PDL_UINT64)
TYPEMAP_STRINGFEATURES_IN(float32_t,     PDL_FLOAT32)
TYPEMAP_STRINGFEATURES_IN(float64_t,     PDL_FLOAT64)
TYPEMAP_STRINGFEATURES_IN(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_STRINGFEATURES_IN(SV*,		 PDL_OBJECT)

#undef TYPEMAP_STRINGFEATURES_IN

/* output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_OUT(type,typecode)
%typemap(out) shogun::SGStringList<type>
{
  $result = sv_newmortal();
  if(!string_to_pdl($result, $1, typecode))
    SWIG_fail;
  if(!is_piddle($result))
    SWIG_fail;
  argvi++;
}
%enddef

TYPEMAP_STRINGFEATURES_OUT(bool,          PDL_BOOL)
TYPEMAP_STRINGFEATURES_OUT(w_char,        PDL_UNICODE)
TYPEMAP_STRINGFEATURES_OUT(char,          PDL_STRING)
TYPEMAP_STRINGFEATURES_OUT(uint8_t,       PDL_UINT8)
TYPEMAP_STRINGFEATURES_OUT(int16_t,       PDL_INT16)
TYPEMAP_STRINGFEATURES_OUT(uint16_t,      PDL_UINT16)
TYPEMAP_STRINGFEATURES_OUT(int32_t,       PDL_INT32)
TYPEMAP_STRINGFEATURES_OUT(uint32_t,      PDL_UINT32)
TYPEMAP_STRINGFEATURES_OUT(int64_t,       PDL_INT64)
TYPEMAP_STRINGFEATURES_OUT(uint64_t,      PDL_UINT64)
TYPEMAP_STRINGFEATURES_OUT(float32_t,     PDL_FLOAT32)
TYPEMAP_STRINGFEATURES_OUT(float64_t,     PDL_FLOAT64)
TYPEMAP_STRINGFEATURES_OUT(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_STRINGFEATURES_OUT(SV*,		  PDL_OBJECT)
#undef TYPEMAP_STRINGFEATURES_ARGOUT


/* input typemap for Sparse Features */
%define TYPEMAP_SPARSEFEATURES_IN(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGSparseMatrix<type>
{
    $1 = is_pdl_sparse_matrix($input, typecode);
}

%typemap(in) shogun::SGSparseMatrix<type>
{
    if(! spmatrix_from_pdl<type>($1, $input, typecode))
      SWIG_fail;
}
%enddef

TYPEMAP_SPARSEFEATURES_IN(bool,          PDL_BOOL)
TYPEMAP_SPARSEFEATURES_IN(w_char,        PDL_UNICODE)
TYPEMAP_SPARSEFEATURES_IN(char,          PDL_STRING)
TYPEMAP_SPARSEFEATURES_IN(uint8_t,       PDL_UINT8)
TYPEMAP_SPARSEFEATURES_IN(int16_t,       PDL_INT16)
TYPEMAP_SPARSEFEATURES_IN(uint16_t,      PDL_UINT16)
TYPEMAP_SPARSEFEATURES_IN(int32_t,       PDL_INT32)
TYPEMAP_SPARSEFEATURES_IN(uint32_t,      PDL_UINT32)
TYPEMAP_SPARSEFEATURES_IN(int64_t,       PDL_INT64)
TYPEMAP_SPARSEFEATURES_IN(uint64_t,      PDL_UINT64)
TYPEMAP_SPARSEFEATURES_IN(float32_t,     PDL_FLOAT32)
TYPEMAP_SPARSEFEATURES_IN(float64_t,     PDL_FLOAT64)
TYPEMAP_SPARSEFEATURES_IN(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_SPARSEFEATURES_IN(SV*,		 PDL_OBJECT)
#undef TYPEMAP_SPARSEFEATURES_IN

/* output typemap for sparse features returns (data, row, ptr) */
%define TYPEMAP_SPARSEFEATURES_OUT(type,typecode)
%typemap(out) shogun::SGSparseVector<type>
{
  if(!spvector_to_pdl($result, $1, typecode))
    SWIG_fail;
}

%typemap(out) shogun::SGSparseMatrix<type>
{
  $result = sv_newmortal();
  if(!spmatrix_to_pdl($result, $1, typecode))
    SWIG_fail;
  if(!is_piddle($result))
    SWIG_fail;
  argvi++;
}
%enddef

TYPEMAP_SPARSEFEATURES_OUT(bool,          PDL_BOOL)
TYPEMAP_SPARSEFEATURES_OUT(w_char,        PDL_UNICODE)
TYPEMAP_SPARSEFEATURES_OUT(char,          PDL_STRING)
TYPEMAP_SPARSEFEATURES_OUT(uint8_t,       PDL_UINT8)
TYPEMAP_SPARSEFEATURES_OUT(int16_t,       PDL_INT16)
TYPEMAP_SPARSEFEATURES_OUT(uint16_t,      PDL_UINT16)
TYPEMAP_SPARSEFEATURES_OUT(int32_t,       PDL_INT32)
TYPEMAP_SPARSEFEATURES_OUT(uint32_t,      PDL_UINT32)
TYPEMAP_SPARSEFEATURES_OUT(int64_t,       PDL_INT64)
TYPEMAP_SPARSEFEATURES_OUT(uint64_t,      PDL_UINT64)
TYPEMAP_SPARSEFEATURES_OUT(float32_t,     PDL_FLOAT32)
TYPEMAP_SPARSEFEATURES_OUT(float64_t,     PDL_FLOAT64)
TYPEMAP_SPARSEFEATURES_OUT(floatmax_t,    PDL_LONGDOUBLE)
TYPEMAP_SPARSEFEATURES_OUT(SV*,		  PDL_OBJECT)

#undef TYPEMAP_SPARSEFEATURES_OUT
#endif /* HAVE_PDL */
