/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This code is inspired by the perl data language package typemaps
 *
 * Written (W) 2012 Christian Montanari
 */


//%constant   enum boolean {True=true, False=false};

%constant  bool True = true;
%constant  bool False = false;

%{
extern "C" {
#include <pdl.h>
#include <pdlcore.h>
#include <values.h>
}
    //PTZ121012 from PDL
#define MAX_DIMENSIONS 100
#define MAX_VAR_DIMS 32
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

    /* Functions to extract array attributes.
     */
    //TODO::PTZ121113 not completly right since Hash object will also be detected
    //, better use magic, SvPOK(a)
        static pdl* if_piddle(SV* a) {
            pdl* it = 0;
            if(SvROK(a) && ((SvTYPE(SvRV(a)) == SVt_PVMG) || (SvTYPE(SvRV(a)) == SVt_PVHV))) {
                it = PDL->SvPDLV(a);
            }
            return it;
        }


        static int array_type(SV* sv) {
            pdl* it = PDL->SvPDLV(sv);
            return it->datatype;
        }

        static bool is_array(SV* a) {
            return SVavref(a);
        }

        static bool is_piddle(SV* a) {
            return(if_piddle(a) != 0 ? true : false);
        }

        static int array_dimensions(SV* sv) {
            pdl* it = if_piddle(sv);
            if(it->state & PDL_NOMYDIMS) {
                return -1; //NODIMS?
            }
            return it->ndims;
        }

        //
        static size_t array_size(SV* sv, int i) {
            pdl* x = if_piddle(sv);
            pdl_make_physvaffine( x );
            //size_t *incs = (PDL_VAFFOK(x) ? x->vafftrans->incs : x->dimincs);
            //return it->dimincs;
            return x->nvals;
        }

        static bool array_is_contiguous(SV* sv) {
            //PTZ120930 can we call XS_PDL_iscontig(sv); line 342 "Core.xs"
            pdl* x =  PDL->SvPDLV(sv);
            int RETVAL = true;
            pdl_make_physvaffine( x );
            if PDL_VAFFOK(x) {
                int i, inc=1;
                printf("vaff check...\n");
                for (i=0;i<x->ndims;i++) {
                    if (PDL_REPRINC(x,i) != inc) {
                        RETVAL = false;
                        break;
                    }
                    inc *= x->dims[i];
                }
            }
            //  pdl* it = SvPDLV(sv);
            //  return (it->state & (PDL_ALLOCATED | PDL_HDRCPY));
            //PTZ120927 needs to check... in
            return RETVAL;
        }


        static bool is_pdl_narry(SV* sv, int ndims_min, int ndims_max)
        {
            pdl* it = if_piddle(sv);
            if(it != 0) {
                pdl_make_physdims(it);
                if((!(it->state & PDL_NOMYDIMS))
                        && it->ndims >= ndims_min
                        && it->ndims < ndims_max) {
                    return true;
                }
            } else if(is_array(sv)) {
                return true;
            }
            return false;
        }

        static int is_pdl_vector(SV* sv, int typecode)
        {
            return(is_pdl_narry(sv, 1, 2));
        }

        static bool is_swig_this_or_piddle_vector(SV* a, int typecode, swig_type_info *_t)
	{
	    void *vptr = 0;
	    int res = SWIG_ConvertPtr(a, &vptr, _t, 0);
	    if(SWIG_CheckState(res)) {
	      return true;
	    } else {
	      return(is_pdl_narry(a, 1, 2));
	    }
        }

        /* check  for rectangular matrix (2D) */
        static int is_pdl_matrix(SV* sv, int typecode)
        {
            return(is_pdl_narry(sv, 2, 3));
        }

        /* check  for PDL (rectangular) array N-ary */
        static int is_pdl_array(SV* sv, int typecode)
        {
            return(is_pdl_narry(sv, 3, MAX_DIMENSIONS));
        }

        /* check  for sparse? matrix (2D)  */
        static int is_pdl_sparse_matrix(SV* sv, int typecode)
        {
            return(array_is_contiguous(sv) && is_pdl_matrix(sv, typecode));
#if 0
            return ( obj && SV_HasAttrString(obj, "indptr") &&
                    SV_HasAttrString(obj, "indices") &&
                    SV_HasAttrString(obj, "data") &&
                    SV_HasAttrString(obj, "shape")
                   ) ? 1 : 0;
#endif
        }

        //PTZ121011... check dims >= 2 and dims[-1] == 1
        static int is_pdl_string(SV* sv, int typecode)
        {
            pdl* it = if_piddle(sv);
            if(it != 0) {
                return (it->datatype == typecode);
            }
            return false;
        }

        static int is_pdl_string_list(SV* sv, int typecode)
        {
            pdl* it = PDL->SvPDLV(sv);
            return (it->datatype == typecode);
            //PTZ120927 for now

#if 0
            SV* list=(SV*) obj;
            int result=0;
            if (list && PyList_Check(list) && PyList_Size(list)>0)
            {
                result=1;
                int32_t size=PyList_Size(list);
                for (int32_t i=0; i<size; i++)
                {
                    SV *o = PyList_GetItem(list,i);
                    if (typecode == NPY_STRING || typecode == NPY_UNICODE)
                    {
                        if (!PyString_Check(o))
                        {
                            result=0;
                            break;
                        }
                    }
                    else
                    {
                        if (!is_array(o) || array_dimensions(o)!=1 || array_type(o) != typecode)
                        {
                            result=0;
                            break;
                        }
                    }
                }
            }
            return result;
#endif
        }




        //form PDL/Basic/Core/Core.xs
#define XS_PDL_setflag(reg,flagval,val) (val?(reg |= flagval):(reg &= ~flagval))



        /* this is horrible - the routines from bad should perhaps be here instead ? */
        double pdl_get_badvalue( int datatype ) {
            double retval;
            switch ( datatype ) {
                case PDL_B: retval = PDL->bvals.Byte; break;
                case PDL_S: retval = PDL->bvals.Short; break;
                case PDL_US: retval = PDL->bvals.Ushort; break;
                case PDL_L: retval = PDL->bvals.Long; break;
                case PDL_LL: retval = PDL->bvals.LongLong; break;
                case PDL_F: retval = PDL->bvals.Float; break;
                case PDL_D: retval = PDL->bvals.Double; break;

                default:
                            croak("Unknown type sent to pdl_get_badvalue\n");
            }
            return retval;
        } /* pdl_get_badvalue() */


        double pdl_get_pdl_badvalue( pdl *it ) {
            double retval;
            int datatype;


            datatype = it->datatype;
            retval = pdl_get_badvalue( datatype );
            return retval;
        } /* pdl_get_pdl_badvalue() */


        //line 23 pdlcore.c
        static SV* getref_pdl(pdl* it) {
            SV* newref;
            if(!it->sv) {
                SV *ref;
                HV *stash = gv_stashpv("PDL",TRUE);
                SV *psv = newSViv(PTR2IV(it));
                it->sv = psv;
                newref =  newRV_noinc((SV*) it->sv);
                (void) sv_bless(newref,stash);
            } else {
                newref = newRV_inc((SV*)it->sv);
                SvAMAGIC_on(newref);
            }
            return newref;
        }

        void pdl_make_scratch_hash(pdl *ret,double data, int datatype) {
            STRLEN n_a;
            HV *hash;
            SV *dat; PDL_Long fake[1];

            /* Compress to smallest available type. This may have strange
               results sometimes :( */
            ret->datatype = datatype;
            ret->data = pdl_malloc(pdl_howbig(ret->datatype)); /* Wasteful */
            //PTZ121004, yo, maybe need data PDL_B?
            dat = newSVpv((char *)ret->data, pdl_howbig(ret->datatype));

            ret->data = SvPV(dat, n_a);
            ret->datasv = dat;

            /* This is an important point: it makes this whole piddle mortal
             * so destruction will happen at the right time.
             * If there are dangling references, pdlapi.c knows not to actually
             * destroy the C struct. */
            sv_2mortal(getref_pdl(ret));

            pdl_setdims(ret, fake, 0); /* However, there are 0 dims in scalar */
            ret->nvals = 1;

            /* NULLs should be ok because no dimensions. */
            pdl_set(ret->data, ret->datatype, NULL, NULL, NULL, 0, 0, data);
        }


        //PTZ121111 I made it up, not used.
        SV* pdl_core_listref_pv_string(pdl * x)
        {
            PDL_Long * inds;
            PDL_Long * incs;
            size_t offs;
            void *data;
            SV** xx;

            int ind; int lind; int i;
            int stop = 0;
            AV *av;
            SV *sv;
            double pdl_badval;
            int badflag = (x->state & PDL_BADVAL) > 0;
            if ( badflag ) {
                pdl_badval = pdl_get_pdl_badvalue( x );
            }
            pdl_make_physvaffine( x );
            inds = (PDL_Long *) pdl_malloc(sizeof(PDL_Long) * x->ndims); /* GCC -> on stack :( */
            data = PDL_REPRP(x);
            xx = (SV**) data;

            incs = (PDL_VAFFOK(x) ? x->vafftrans->incs : x->dimincs);
            offs = PDL_REPROFFS(x);
            av = newAV();
            av_extend(av, x->nvals);
            lind = 0;
            for(ind = 0; ind < x->ndims; ind++) inds[ind] = 0;
            while(!stop) {
                //SV* pdl_val = cast<SV*>(pdl_at(data, x->datatype, inds, x->dims, incs, offs, x->ndims));
                //cast refused get index and mem
                i = pdl_get_offset(inds, x->dims, incs, offs, x->ndims);
                sv = newSVrv(xx[i], NULL);
                av_store(av, lind, sv);
                lind++;
                stop = 1;
                for(ind = 0; ind < x->ndims; ind++) {
                    if(++(inds[ind]) >= x->dims[ind]) {
                        inds[ind] = 0;
                    } else {
                        stop = 0;
                        break;
                    }
                }
            }
            return  newRV_noinc((SV *)av);
        }
    /*
    //TODO::PTZ121111 there is an unfortunate need to transpose...
    PDL dimension format is reverse to the mathematical conventions (sigh).
     which means a transposition of dimensions and memory....
     so bye bye to  sequencial  memcopy until a genious programmer knows better.
    ie.:
     PDL sequence(3,2) is dim [3,2] and [[0 1 2] [3 4 5]]
    in memory...
    .
    -
    0		<=> pdl->at(0,0) <=> sg->at(0,0)
    -
    1		<=> pdl->at(0,1) <=> sg->at(0,1)
    -
    2		<=> pdl->at(0,2) <=> sg->at(1,0) ...
    -
    ...
    -
    4
    -
    5
    -
    .
    */

    template <class type>
      static bool sg_to_pdl(SV* rsv, type* data_sg, index_t* dims_sg, index_t ndims_sg, int typecode)
        {
	  pdl* it = PDL->pdlnew();
	  if(!it) {
	    return false;
	  }
	  PDL_Long *dims_pdl = (PDL_Long *) pdl_malloc(sizeof(PDL_Long) * ndims_sg);
	  PDL_Long *inds = (PDL_Long *) pdl_malloc(sizeof(PDL_Long) * ndims_sg);
	  if(!inds) {
	      return false;
	  }
	  for(int i = 0; i < ndims_sg; i++) {
	    dims_pdl[ndims_sg - i - 1] = *(dims_sg + i);
	    inds[i] = 0;
	  }
	  PDL->setdims(it, dims_pdl, ndims_sg);

	  it->datatype = typecode;
	  PDL->allocdata(it);
	  type *data_pdl = (type *) PDL_REPRP(it);
	  if(!data_pdl) {
	    PDL->destroy(it);
	    return false;
	  }
	  PDL_Long* incs_pdl = (PDL_VAFFOK(it) ? it->vafftrans->incs : it->dimincs);
	  PDL_Long offs_pdl = PDL_REPROFFS(it);
	  PDL_Long nvals_pdl = it->nvals;

	  type val_sg;
	  int lind = 0;
	  int stop = 0;
	  int i_pdl = 0;
	  while(!stop && lind < nvals_pdl) {
	    val_sg = data_sg[lind];
	    //TODO::PTZ121113 check bad values (nan,inf) according to sign and types!
	    pdl_set(data_pdl, typecode, inds, dims_pdl, incs_pdl, offs_pdl, ndims_sg, val_sg);
	    //TODO::PTZ121113 try to trick pdl_get_offset (in pdlsections.c)
	    lind++;
	    stop = 1;
	    i_pdl = 0;
	    for(int i = ndims_sg - 1; 0 <= i; i--) {
	      if(stop) {
		if(++(inds[i]) >= dims_pdl[i]) {
		  inds[i] = 0;
		} else {
		  stop = 0;
		  break;
		}
	      }
	      //PTZ121111 find a better algo using a stack...or this pdl_affine wizbiz
	      //TODO::PTZ121113 try to trick pdl_get_offset with right offset and dims ... increment
	      //i_pdl += dims_pdl[i] * inds[i];
	    }
	  }
	  PDL->SetSV_PDL(rsv, it);
	  return true;
	}

    template <class type>
        static bool sg_from_pdl(type** data_sg, index_t** dims_sg, index_t* ndims_sg, SV* rsv, int typecode)
        {
            pdl* it = PDL->SvPDLV(rsv);

            PDL->make_physdims(it);
            index_t ndims = it->ndims;
            PDL_Long *dims_pdl = it->dims;
            if(!(it->state && PDL_NOMYDIMS)) {
	      return false;
	    }
	    PDL_Long *inds = (PDL_Long *) pdl_malloc(sizeof(PDL_Long) * ndims);
	    if(inds == 0) {
	      return false;
	    }
	    //PTZ121111 use PDL stuff instead of SG here...
	    *dims_sg = SG_MALLOC(index_t, ndims);
	    if((*dims_sg) == 0) {
	      return false;
	    }
	    for(int i = 0; i < ndims; i++) {
	      *((*dims_sg) + i) = dims_pdl[ndims - i - 1];
	      inds[i] = 0;
	    }

            //from set_datatype
            PDL->make_physical(it);
	    if(!PDL_ENSURE_ALLOCATED(it)) {
	      warn("could not allocate PDL (rectangular) matrix memory");
	      return false;
            }
	    if(it->trans) {
                pdl_destroytransform(it->trans, 1);
            }

	    //PTZ121111_TMP?
            pdl_converttype(&it, typecode, PDL_PERM);
            //pdl_make_physvaffine(it);

	    void* data_pdl = PDL_REPRP(it);
	    PDL_Long* incs_pdl = (PDL_VAFFOK(it) ? it->vafftrans->incs : it->dimincs);
            PDL_Long offs_pdl = PDL_REPROFFS(it);
            PDL_Long nvals_pdl = it->nvals;

	    //PTZ121111 use PDL stuff instead of SG here...
            *data_sg = SG_MALLOC(type, nvals_pdl);
	    if((*data_sg) == 0) {
	      return false;
	    }

	    int badflag = (it->state & PDL_BADVAL) > 0;
            type pdl_badval;
            if(badflag) {
                pdl_badval = pdl_get_pdl_badvalue(it);
            }

	    type pdl_val;
            int lind = 0;
            int stop = 0;
            while(!stop && lind < nvals_pdl) {
	      //TODO::PTZ121113 still, try to trick pdl_get_offset (in pdlsections.c)
	      pdl_val = pdl_at(data_pdl, typecode, inds, dims_pdl, incs_pdl, offs_pdl, ndims);
	      if(badflag && pdl_val == pdl_badval) {
		  //TODO::PTZ121111 inf, nan handling
		  //sv = newSVpvn( "BAD", 3 );
		  //change it to a NON_ variable INFTY, MAX_REAL, MACHINE_EPSILON;ALMOST_INFTY;
		  //pdl_val is double
		  // pdl_val = NAN;
	      }
	      *((*data_sg) + lind) = pdl_val;
	      lind++;
	      stop = 1;
	      for(int i = ndims - 1; 0 <= i; i--) {
		if(++(inds[i]) >= dims_pdl[i]) {
		  inds[i] = 0;
		} else {
		  stop = 0;
		  break;
		}
	      }
            }
	    if(lind == nvals_pdl) {
	      *ndims_sg = ndims;
	      return true;
	    }
	    return false;
        }
%}

%typecheck(SWIG_TYPECHECK_POINTER) pdl* {
  $1 = is_piddle($input);
}

%typemap(in) pdl* {
  $result = PDL->SvPDLV($1)
}

%typemap(out) pdl* {
  PDL->SetSV_PDL($result, $1);
}
