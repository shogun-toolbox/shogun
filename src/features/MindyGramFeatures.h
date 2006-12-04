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

		/**
		 * Constructor for word features extracted from string features
		 * @param sf String features to use
		 * @param aname Alphabet name, e.g. bytes, ascii, text, dna
		 * @param delim Escaped string of delimiters, e.g. '%20.,'
		 * @param len   Length of byte array
		 */
		template <class T>
		CMindyGramFeatures(CStringFeatures<T> *sf, CHAR *aname, CHAR *embed, CHAR *delim) : CFeatures(0)
		{
			ASSERT(sf && aname && embed && delim);

			/* Allocate and generate gram configuration (words) */
			CIO::message(M_DEBUG, "Initializing Mindy gram features\n");    
			alph_type_t at = alph_get_type(aname);
			cfg = gram_cfg_words(alph_create(at), delim);
			set_embedding(cfg, embed);

			CIO::message(M_INFO, "Mindy in word mode (d: '%s', a: %s, e: %s)\n", 
					delim, alph_get_name(at), gram_cfg_get_embed(cfg->embed));

			import<T>(sf);
		}

		/**
		 * Constructor for n-gram features extracted from string features
		 * @param sf String feature objects
		 * @param aname Alphabet name, e.g. bytes, ascii, text, dna
		 * @param nlen N-gram length
		 */
		template<class T>
		CMindyGramFeatures(CStringFeatures<T> *sf, CHAR * aname, CHAR * embed, BYTE nlen) : CFeatures(0)
		{
			ASSERT(sf && aname && embed && nlen > 0);

			/* Allocate and generate gram configuration (n-grams) */
			CIO::message(M_DEBUG, "Initializing Mindy gram features\n");
			alph_type_t at = alph_get_type(aname);
			cfg = gram_cfg_ngrams(alph_create(at), (byte_t) nlen);
			set_embedding(cfg, embed);    

			CIO::message(M_INFO, "Mindy in n-gram mode (n: %d, a: %s, e: %s)\n", 
					nlen, alph_get_name(at), gram_cfg_get_embed(cfg->embed));

			import<T>(sf);
		}


		/**
		 * Copy constructor for gram features
		 * @param orig Gram feature object to copy
		 */
		 template <class T>
		 CMindyGramFeatures(const CMindyGramFeatures & orig) : CFeatures(orig)
		 {
		         CIO::message(M_DEBUG, "Duplicating Mindy gram features\n");
		         num_vectors = orig.num_vectors;

		         /* Clone configuration */
		         cfg = gram_cfg_clone(orig.cfg);

		         /* Clone gram vectors */
		         vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
		         for (INT i = 0; i < num_vectors; i++)
		                 vectors[i] = gram_clone(orig.vectors[i]);
                }

		/**
		 * Imports gram features from a string feature object
		 * @param sf String feature object
		 * @return true on success, false otherwise
		 */
		template <class T>
		bool import(CStringFeatures<T> *sf)
		{
			INT i;
			num_vectors = sf->get_num_vectors();
			CIO::message(M_INFO, "Importing %ld string features\n", num_vectors);

			vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
			if (!vectors) {
#ifdef HAVE_PYTHON
            throw FeatureException("Could not allocate memory\n");
#else
				CIO::message(M_ERROR, "Could not allocate memory\n");
#endif
				return false;
			}

			for (i = 0; i < num_vectors; i++) {
				INT len;
				T *s = sf->get_feature_vector(i, len);
				vectors[i] = gram_extract(cfg, (byte_t *) s, (size_t) len);

				CIO::message(M_DEBUG, "Extracted gram vector %d: %d grams\n", i, 
						vectors[i]->num);
			}

			return true;
		}

        /* Destructors */
        ~CMindyGramFeatures();

        CFeatures *duplicate() const;

        /* Feature and vector functions */
        void set_embedding(gram_cfg_t *, CHAR *);
        gram_t *get_feature_vector(INT i);
        void set_feature_vector(INT i, gram_t * g);
        inline ULONG get_feature(INT i, INT j);
        inline INT get_vector_length(INT i);

        /* Simple functions */
        virtual inline INT get_num_vectors() { return num_vectors; }
        virtual inline INT get_size() { return sizeof(gram_t *); }
        EFeatureClass get_feature_class() { return C_MINDYGRAM; }
        EFeatureType get_feature_type() { return F_ULONG; }

    protected:

        /* Import and load functions */
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
