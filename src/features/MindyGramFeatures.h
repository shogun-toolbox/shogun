/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Written (W) 2006-2008 Soeren Sonnenburg
 * Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
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
class CMindyGramFeatures : public CFeatures
{
    public:
		/**
		 * Constructor for word features extracted from string features
		 * @param aname Alphabet name, e.g. bytes, ascii, text, dna
		 * @param embed Embedding mode, freq, count, bin
		 * @param delim Escaped string of delimiters, e.g. '%20.,'
		 * @param nlen  K-gram length, 0 = word mode
		 */
		CMindyGramFeatures(char *aname, char *embed, char *delim, uint8_t nlen)
		: CFeatures(0)
		{
			ASSERT(aname && embed && delim);

			/* Allocate and generate gram configuration (words) */
			SG_DEBUG( "Initializing Mindy gram features\n");
			if (nlen == 0)
				cfg = micfg_words(alph_get_type(aname), delim);
			else
				cfg = micfg_ngrams(alph_get_type(aname), (byte_t) nlen);

			/* Set delimiters */
			if (strlen(delim) > 0)
				micfg_set_delim(cfg, delim);
			
			/* Set embedding */
			if (!strcasecmp(embed, "freq"))
				micfg_set_embed(cfg, ME_FREQ);
			else if (!strcasecmp(embed, "count"))
				micfg_set_embed(cfg, ME_COUNT);
			else if (!strcasecmp(embed, "bin"))
				micfg_set_embed(cfg, ME_BIN);
			else
				SG_ERROR("Unknown embedding mode '%s'", embed);

			if (nlen == 0)
			   SG_INFO("Mindy in word mode (d: '%s', a: %s, e: %s)\n",
					   delim, aname, micfg_get_embed(cfg->embed));
			else  
			   SG_INFO("Mindy in n-gram mode (n: '%d', a: %s, e: %s)\n",
					   nlen, aname, micfg_get_embed(cfg->embed));
		}

		/**
		 * Copy constructor for gram features
		 * @param orig Gram feature object to copy
		 */
#if 0
		 CMindyGramFeatures(const CMindyGramFeatures & orig) : CFeatures(orig)
		 {
		         SG_DEBUG( "Duplicating Mindy gram features\n");
		         num_vectors = orig.num_vectors;

		         /* Clone configuration */
		         cfg = micfg_clone(orig.cfg);

		         /* Clone gram vectors */
		         vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
		         for (int32_t i = 0; i < num_vectors; i++)
		                 vectors[i] = gram_clone(orig.vectors[i]);
                }
#endif

		/**
		 * Imports gram features from a string feature object
		 * @param sf String feature object
		 * @return true on success, false otherwise
		 */
		template <class T> 
		bool import_features(CStringFeatures<T> *sf)
		{
			int32_t i;
			num_vectors = sf->get_num_vectors();
			SG_INFO( "Importing %ld string features\n", num_vectors);

			vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
			if (!vectors) {
				SG_ERROR( "Could not allocate memory\n");
				return false;
			}

			for (i = 0; i < num_vectors; i++) {
				int32_t len;
				T *s = sf->get_feature_vector(i, len);
				vectors[i] = gram_extract(cfg, (byte_t *) s, (size_t) len);

				SG_DEBUG( "Extracted gram vector %d: %d grams\n", i, 
						vectors[i]->num);
			}

			return true;
		}

        /* Destructors */
        ~CMindyGramFeatures();

        CFeatures *duplicate() const;

        /* Feature and vector functions */
        gram_t *get_feature_vector(int32_t i);
        void set_feature_vector(int32_t i, gram_t * g);
        uint64_t get_feature(int32_t i, int32_t j);
        int32_t get_vector_length(int32_t i);
        void trim_max(double m);

        /* Simple functions */
        virtual int32_t get_num_vectors() { return num_vectors; }
        virtual int32_t get_size() { return sizeof(gram_t *); }
        EFeatureClass get_feature_class() { return C_MINDYGRAM; }
        EFeatureType get_feature_type() { return F_ULONG; }

    protected:
        /* Import and load functions */
        virtual bool load(char * fname);

    private:
        /**< number of gram vectors */
        int32_t num_vectors;
        /**< Array of gram features */
        gram_t **vectors;
        /**< Gram configuration used */
        micfg_t *cfg;
};
#endif
#endif
