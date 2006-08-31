/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_MINDY

#ifndef _CMINDYGRAMFEATURES__H__
#define _CMINDYGRAMFEATURES__H__

#include "features/Features.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"

#include <mindy.h>

// MindyGramFeatures
class CMindyGramFeatures:public CFeatures
{

    public:

        /* Constructors */
        CMindyGramFeatures(CHAR * fname, CHAR * aname, BYTE nlen);
        CMindyGramFeatures(CStringFeatures < CHAR > *sf, CHAR * aname, BYTE nlen);
        CMindyGramFeatures(CHAR *fname, CHAR *aname, CHAR *delim);
        CMindyGramFeatures(CStringFeatures<CHAR> *sf, CHAR *aname, CHAR *delim);
        CMindyGramFeatures(const CMindyGramFeatures & orig);
        ~CMindyGramFeatures();

        CFeatures *duplicate() const;

        /* Feature and vector functions */
        gram_t *get_feature_vector(INT i);
        void set_feature_vector(INT i, gram_t * g);
        inline byte_t *get_feature(INT i, INT j);
        inline INT get_vector_length(INT i);

        /* Simple functions */
        virtual inline INT get_num_vectors() { return num_vectors; }
        virtual inline INT get_size() { return sizeof(gram_t *); }
        EFeatureClass get_feature_class() { return C_MINDYGRAM; }
        EFeatureType get_feature_type() { return F_ULONG; }

    protected:

        /* Import and load functions */
        virtual bool import(CStringFeatures < CHAR > *sf);
        virtual bool load(CHAR * fname);

    private:

        /**< number of gram vectors */
        INT num_vectors;
        /**< Array of gram features */
        gram_t **vectors;
        /**< Gram configuration used */
        gram_cfg_t *cfg;
};
#endif
#endif
