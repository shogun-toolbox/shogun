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

#ifdef HAVE_PDL

%include pdl.i

%{
#include <lib/DataType.h>

    /* OUT typemaps */
    /****************/
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
            void *data = PDL_REPRP(it);
            if(!data) {
                PDL->destroy(it);
                return false;
            }
            memcpy((type *)data, sg_vec.vector, clen);
            PDL->SetSV_PDL(rsv, it);
            return true;
        }

    template <class type>
      static bool matrix_to_pdl(SV* rsv, SGMatrix< type > matrix_sg, int typecode)
      {
	index_t dims_sg[2] = {matrix_sg.num_rows, matrix_sg.num_cols};
	bool e = sg_to_pdl(rsv, matrix_sg.matrix, dims_sg, 2, typecode);
	return e;
      }

    template <class type>
      static bool array_to_pdl(SV* rsv, SGNDArray< type > array_sg, int typecode)
      {
	bool e = sg_to_pdl(rsv, array_sg.array, array_sg.dims, array_sg.num_dims, typecode);
	return e;
      }

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






    /* IN typemaps */
    /***************/

    template <class type>
        static bool vector_from_pdl(SGVector<type>& sg_vec, SV* obj, int typecode)
        {
            pdl* it = PDL->SvPDLV(obj);
            it = PDL->get_convertedpdl(it, typecode);
            PDL->make_physical(it);
            if(!PDL_ENSURE_ALLOCATED(it)) {
                warn("could not allocate PDL vector memory");
                return false;
            }
            void* data = get_copy(PDL_REPRP(it), sizeof(type) * it->nvals);
            sg_vec = SGVector<type>((type *)data, it->dims[0]);
            return true;
        }

    template <class type>
        static bool matrix_from_pdl(SGMatrix<type>& matrix_sg, SV* rsv, int typecode)
        {
	  type* data_sg = 0;
	  index_t* dims_sg = 0;
	  index_t ndims_sg = 0;
	  bool e = sg_from_pdl< type >(&data_sg, &dims_sg, &ndims_sg, rsv, typecode);
	  if(e) {
	    matrix_sg = SGMatrix< type > (data_sg, dims_sg[0],  dims_sg[1], true);
	    return true;
	  }
	  if(dims_sg) {SG_FREE(dims_sg);}
	  if(data_sg) {SG_FREE(data_sg);}
	  return false;
	}

    template < class type >
        static bool array_from_pdl(SGNDArray<type>& array_sg, SV* rsv, int typecode)
        {
	  type* data_sg = 0;
	  index_t* dims_sg = 0;
	  index_t ndims_sg = 0;
	  bool e = sg_from_pdl< type >(&data_sg, &dims_sg, &ndims_sg, rsv, typecode);
	  if(e) {
	    array_sg = SGNDArray< type >(data_sg, dims_sg, ndims_sg);
	    return true;
	  }
	  if(dims_sg) {SG_FREE(dims_sg);}
	  if(data_sg) {SG_FREE(data_sg);}
	  return false;
	}
    /* string stuff */

        /*
         * typically, a PDL::Char dimension will be....
         * ...(strlen~columns)_ x rows_ x twist_ x .... x (size_t)_
         * and sequencialy set so in memory:
	 * string_[.0xxx0]....string_[.nxxxn]
         *
         */
    //TODO::PTZ121114 really to put into pdl.i
    template <class type>
      static bool string_from_pdl(SGStringList< type >& sg_strings, SV* sv, U32 typecode)
            {
                pdl* it = if_piddle(sv);
                if(it) {
                    it = pdl_get_convertedpdl(it, typecode);
                    PDL->make_physical(it); /* Wasteful*/
                    if(!PDL_ENSURE_ALLOCATED(it)) {
                        pdl_dump(it); /* never happens */
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
                    SGString< type >* l_ss = SG_MALLOC(SGString< type >, l_sz);
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
			//TODO::PTZ121114 nvals is always in char unit, what happens when short unit (unicode)?
			el_str = (char*) data + i;
                        el_len = strnlen(el_str, l_len_max);
                        if(i + el_len > it->nvals) {
                            warn("string_from_pdl::offset error in string conversion::bayling out");
                            for(int32_t j = 0; j < lind; j++)
                                l_ss[j].~SGString< type >();
                            SG_FREE(l_ss);
                            //free(inds);
                            return false;
                        }
                        new(&l_ss[lind]) SGString< type >();
                        l_ss[lind].slen = el_len;
                        l_ss[lind].string = NULL;
                        if(el_len > 0) {
                            l_ss[lind].string = SG_MALLOC(type, el_len);
                            if(!l_ss[lind].string) {
			      warn("string_from_pdl::out of memory");
			      for(int32_t j = 0; j <= lind; j++)
                                    l_ss[j].~SGString< type >();
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
			for(int n = ndims - 1; 0 < n; n--) {
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

    //TODO:PTZ121114 later...
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
%typecheck(SWIG_TYPECHECK_POINTER) shogun::SGVector< type >
{
  void * v_p = 0;
  int res = SWIG_ConvertPtr($input, &v_p, $&1_descriptor, 0);
  if(SWIG_CheckState(res)) {
    $1 = true;
  } else {
    $1 = is_pdl_vector($input, typecode);
  }
}
%typemap(in) shogun::SGVector< type >
{
  void * v_p = 0;
  int res = SWIG_ConvertPtr($input, &v_p, $&1_descriptor, 0);
  if(SWIG_IsOK(res) && v_p) {
    $1 = *(reinterpret_cast< shogun::SGVector< type > * >(v_p));
  } else if(!vector_from_pdl< type >($1, $input, typecode)) {
    SWIG_fail;
  }
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
  SWIG_Object r = VOID_Object;
  if(!vector_to_pdl(r, $1, typecode)) SWIG_fail;
  if(!is_piddle(r)) SWIG_fail;
  %set_output(r);
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
%typecheck(SWIG_TYPECHECK_POINTER) shogun::SGMatrix< type >
{
  void * m_p = 0;
  int res = SWIG_ConvertPtr($input, &m_p, $&1_descriptor, 0);
  if(SWIG_CheckState(res)) {
    $1 = true;
  } else {
    $1 = is_pdl_matrix($input, typecode);
  }
}
%typemap(in) shogun::SGMatrix< type >
{
  void * m_p = 0;
  int res = SWIG_ConvertPtr($input, &m_p, $&1_descriptor, 0);
  if(SWIG_IsOK(res) && m_p) {
    $1 = *(reinterpret_cast< shogun::SGMatrix< type > * >(m_p));
  } else if(!matrix_from_pdl< type >($1, $input, typecode)) {
    SWIG_fail;
  }
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
  SWIG_Object r = VOID_Object;
  if(!matrix_to_pdl(r, $1, typecode)) SWIG_fail;
  if(!is_piddle(r)) SWIG_fail;
  %set_output(r);
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
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGNDArray<type>
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
%define TYPEMAP_STRINGFEATURES_IN(type, typecode)
%typecheck(SWIG_TYPECHECK_POINTER) shogun::SGStringList< type >
{
  void * s_p = 0;
  int res = SWIG_ConvertPtr($input, &s_p, $&1_descriptor, 0);
  if(SWIG_CheckState(res)) {
    $1 = true;
  } else {
    $1 = is_pdl_string($input, typecode);
  }
}
%typemap(in) shogun::SGStringList< type >
{
  void * s_p = 0;
  int res = SWIG_ConvertPtr($input, &s_p, $&1_descriptor, 0);
  if(SWIG_IsOK(res) && s_p) {
    $1 = *(reinterpret_cast< shogun::SGStringList< type > * >(s_p));
  } else if(!string_from_pdl< type >($1, $input, typecode)) {
    SWIG_fail;
  }
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
%define TYPEMAP_STRINGFEATURES_OUT(type, typecode)
%typemap(out) shogun::SGStringList<type>
{
  SWIG_Object r = VOID_Object;
  if(!string_to_pdl(r, $1, typecode)) SWIG_fail;
  if(!is_piddle(r)) SWIG_fail;
  %set_output(r);
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
