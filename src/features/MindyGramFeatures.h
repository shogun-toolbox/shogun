/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifdef HAVE_MINDY

#ifndef _CMINDYGRAMFEATURES__H__
#define _CMINDYGRAMFEATURES__H__

#include "features/Features.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"

#include <math.h>
#include <mindy.h>

// MindyGramFeatures
class CMindyGramFeatures:public CFeatures {

  public:

    /**
     * Constructor for n-gram features extracted from a text file
     * @param fname File name to load data from
     * @param aname Alphabet name, e.g. bytes, ascii, text, dna
     * @param nlen N-gram length
     */
    CMindyGramFeatures(CHAR * fname, CHAR * aname, BYTE nlen)
        :CFeatures(fname) {
        ASSERT(fname && aname && nlen > 0);

        /* Allocate and generate gram configuration (n-grams) */
        CIO::message(M_DEBUG, "Creating alphabet %s\n", aname);
        alph_type_t at = alph_get_type(aname);
        CIO::message(M_DEBUG, "Creating %d-gram configuration\n", nlen);
        cfg = gram_cfg_ngrams(alph_create(at), (byte_t) nlen);

        CIO::message(M_DEBUG, "Loading data file %s\n", fname);
        load(fname);
    }
    
    /**
     * Constructor for n-gram features extracted from string features
     * @param sf String feature objects
     * @param aname Alphabet name, e.g. bytes, ascii, text, dna
     * @param nlen N-gram length
     */
    CMindyGramFeatures(CStringFeatures < CHAR > *sf, CHAR * aname, BYTE nlen) 
        : CFeatures(0l) {
        ASSERT(aname && nlen > 0);

        /* Allocate and generate gram configuration (n-grams) */
        CIO::message(M_DEBUG, "Creating alphabet %s\n", aname);
        alph_type_t at = alph_get_type(aname);
        CIO::message(M_DEBUG, "Creating %d-gram configuration\n", nlen);
        cfg = gram_cfg_ngrams(alph_create(at), (byte_t) nlen);

        import(sf);
    }

    /**
     * Constructor for word features extracted from a text file
     * @param fname Filen ame to load data from
     * @param aname Alphabet name, e.g. bytes, ascii, text, dna
     * @param delim Escaped string of delimiters, e.g. '%20.,'
     */
    CMindyGramFeatures(CHAR *fname, CHAR *aname, CHAR *delim)
        : CFeatures(fname) {
        ASSERT(fname && aname && delim);

        /* Allocate and generate gram configuration (words) */
        CIO::message(M_DEBUG, "Creating alphabet %s\n", aname);
        alph_type_t at = alph_get_type(aname);
        CIO::message(M_DEBUG, "Creating word configuration\n");
        cfg = gram_cfg_words(alph_create(at), delim);

        load(fname);
    }

    /**
     * Constructor for word features extracted from string features
     * @param sf String features to use
     * @param aname Alphabet name, e.g. bytes, ascii, text, dna
     * @param delim Escaped string of delimiters, e.g. '%20.,'
     * @param len   Length of byte array
     */
    CMindyGramFeatures(CStringFeatures<CHAR> *sf, CHAR *aname, CHAR *delim) 
        : CFeatures(0l) {
        ASSERT(aname && delim);

        /* Allocate and generate gram configuration (words) */
        CIO::message(M_DEBUG, "Creating alphabet %s\n", aname);
        alph_type_t at = alph_get_type(aname);
        CIO::message(M_DEBUG, "Creating word configuration\n");
        cfg = gram_cfg_words(alph_create(at), delim);

        import(sf);
    }


    /** 
     * Copy constructor for gram features
     * @param orig Gram feature object to copy
     */
    CMindyGramFeatures(const CMindyGramFeatures & orig)
        :CFeatures(orig) {
        CIO::message(M_DEBUG, "Duplicating mindy gram features\n");
        num_vectors = orig.num_vectors;

        /* Clone configuration */
        cfg = gram_cfg_clone(orig.cfg);

        /* Clone gram vectors */
        vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
        for (INT i = 0; i < num_vectors; i++)
            vectors[i] = gram_clone(orig.vectors[i]);
    }

    /** 
     * Destructor for gram features
     */
    ~CMindyGramFeatures() {
        CIO::message(M_DEBUG, "Destroying mindy gram features\n");
        /* Destroy gram vectors */
        for (INT i = 0; i < num_vectors; i++)
            gram_destroy(vectors[i]);
        free(vectors);

        /* Destroy configuration */
        alph_destroy(cfg->alph);
        gram_cfg_destroy(cfg);
    }

    /**
     * Duplicate a gram feature object 
     */
    CFeatures *duplicate() const {
        return new CMindyGramFeatures(*this);
    }
    
    /** 
     * Get gram vector for sample i
     * @param i index of gram vector
     * @return gram vector
     */ 
    gram_t *get_feature_vector(INT i) {
        ASSERT(vectors != NULL);
        ASSERT(i >= 0 && i < num_vectors);

        return vectors[i];
    }

    /** 
     * Set gram vector for sample i
     * @param num index of feature vector
     */
    void set_feature_vector(INT i, gram_t * g) {
        ASSERT(vectors != NULL);
        ASSERT(i >= 0 && i < num_vectors);

        /* Destroy previous gram */
        if (vectors[i])
            gram_destroy(vectors[i]);

        vectors[i] = g;
    }

    /**
     * Get a feature (gram) from a gram vector
     * @param i Index of gram vector
     * @param j Index of feature in gram vector
     * @param b Buffer to hold gram of at least 65 bytes
     * @return gram (e.g. an n-gram or word)
     */
    inline byte_t *get_feature(INT i, INT j) {
        ASSERT(vectors && i < num_vectors);
        ASSERT(j < (signed) vectors[i]->num);

        return gram_restore(vectors[i]->gram[j], cfg);
    }

    /**
     * Get the length of gram vector at index i
     * @param i Index of gram vector
     * @return length of gram vector 
     */
    inline INT get_vector_length(INT i) {
        ASSERT(vectors && i < num_vectors);
        return vectors[i]->num;
    }

    /**
     * Get number of vectors
     * @return number of gram vectors
     */
    inline INT get_num_vectors() {
        return num_vectors;
    }

    /**
     * Get size of one gram entity
     * @return size of gram pointer 
     */
    virtual INT get_size() {
        return sizeof(gram_t *);
    }

    /** 
     * Get the feature class
     */
    EFeatureClass get_feature_class() {
        return C_MINDYGRAM;
    }

    /**
     * Get the feature type
     */
    EFeatureType get_feature_type() {
        return F_ULONG;
    }

    private:

    /**
     * Imports gram features from a string feature object
     * @param sf String feature object
     * @return true on success, false otherwise 
     */
    bool import(CStringFeatures < CHAR > *sf) {
        INT i;
        num_vectors = sf->get_num_vectors();
        CIO::message(M_INFO, "Importing %ld string features\n", num_vectors);

        vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
        if (!vectors) {
            CIO::message(M_ERROR, "could not allocate memory\n");
            return false;
        }

        for (i = 0; i < num_vectors; i++) {
            INT len;
            CHAR *s = sf->get_feature_vector(i, len);
            vectors[i] = gram_extract(cfg, (byte_t *) s, (size_t) len);
            
            CIO::message(M_DEBUG, "Vector %d: %d grams\n", i, 
                         vectors[i]->num);
        }

        return true;
    }

    /**
     * Loads a set of strings and extracts corresponding gram vectors
     * @param fname File name 
     * @return true on success, false otherwise 
     */
    virtual bool load(CHAR * fname) {
        CIO::message(M_INFO, "loading...\n");
        LONG len = 0;
        CHAR *s, *t;

        CFile f(fname, 'r', F_CHAR);
        CHAR *data = f.load_char_data(NULL, len);

        if (!f.is_ok()) {
            CIO::message(M_ERROR, "reading file failed\n");
            return false;
        }

        /* Count strings terminated by \n */
        num_vectors = 0;
        for (LONG i = 0; i < len; i++)
            if (data[i] == '\n')
                CIO::message(M_INFO, "file contains %ld vectors\n",
                             num_vectors);

        vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
        if (!vectors) {
            CIO::message(M_ERROR, "could not allocate memory\n");
            return false;
        }

        /* Extract grams from strings */
        t = s = data;
        for (LONG i = 0; i < num_vectors; i++, t++) {
            if (*t != '\n')
                continue;

            vectors[i] = gram_extract(cfg, (byte_t *) s, t - s);
            s = t + 1;
        }

        return true;
    }

    protected:

    /**< number of gram vectors */
    INT num_vectors;
    /**< Array of gram features */
    gram_t **vectors;
    /**< Gram configuration used */
    gram_cfg_t *cfg;
};

#endif
#endif
