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
#include  <pdl.h>
#include <pdlcore.h>
#include  <values.h>
}
    //PTZ121012 from PDL
#define MAX_DIMENSIONS 100
#define MAX_VAR_DIMS 32

    /* Functions to extract array attributes.
     */

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

        //check  for rectangular matrix (2D)
        static int is_pdl_matrix(SV* sv, int typecode)
        {
            return(is_pdl_narry(sv, 2, 3));
        }

        //check  for PDL (rectangular) array N-ary
        static int is_pdl_array(SV* sv, int typecode)
        {
            return(is_pdl_narry(sv, 3, MAX_DIMENSIONS));
        }

        //check  for sparse? matrix (2D) 
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


        //I made it up
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
