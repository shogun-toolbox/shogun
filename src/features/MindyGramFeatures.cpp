/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 *
 * Indentation: bcpp -f 1 -s -ylcnc -bcl -i 4
 */

#include "lib/config.h"

#ifdef HAVE_MINDY

#include "features/Features.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "features/MindyGramFeatures.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"

#include <math.h>
#include <mindy.h>

/**
 * Constructor for n-gram features extracted from a text file
 * @param fname File name to load data from
 * @param aname Alphabet name, e.g. bytes, ascii, text, dna
 * @param nlen N-gram length
 */
CMindyGramFeatures::CMindyGramFeatures(CHAR * fname, CHAR * aname, CHAR * embed, BYTE nlen) : CFeatures(fname)
{
    ASSERT(fname && aname && embed && nlen > 0);

    /* Allocate and generate gram configuration (n-grams) */
    CIO::message(M_DEBUG, "Initializing Mindy gram features\n");    
    alph_type_t at = alph_get_type(aname);
    cfg = gram_cfg_ngrams(alph_create(at), (byte_t) nlen);
    set_embedding(cfg, embed);

    CIO::message(M_INFO, "Mindy in n-gram mode (n: %d, a: %s, e: %s)\n", 
                 nlen, alph_get_name(at), gram_cfg_get_embed(cfg->embed));

    load(fname);
}

/**
 * Constructor for n-gram features extracted from string features
 * @param sf String feature objects
 * @param aname Alphabet name, e.g. bytes, ascii, text, dna
 * @param nlen N-gram length
 */
CMindyGramFeatures::CMindyGramFeatures(CStringFeatures < CHAR > *sf, CHAR * aname, CHAR * embed, BYTE nlen) : CFeatures(0)
{
    ASSERT(sf && aname && embed && nlen > 0);

    /* Allocate and generate gram configuration (n-grams) */
    CIO::message(M_DEBUG, "Initializing Mindy gram features\n");
    alph_type_t at = alph_get_type(aname);
    cfg = gram_cfg_ngrams(alph_create(at), (byte_t) nlen);
    set_embedding(cfg, embed);    

    CIO::message(M_INFO, "Mindy in n-gram mode (n: %d, a: %s, e: %s)\n", 
                 nlen, alph_get_name(at), gram_cfg_get_embed(cfg->embed));

    import(sf);
}

/**
 * Constructor for word features extracted from a text file
 * @param fname Filen ame to load data from
 * @param aname Alphabet name, e.g. bytes, ascii, text, dna
 * @param delim Escaped string of delimiters, e.g. '%20.,'
 */
CMindyGramFeatures::CMindyGramFeatures(CHAR *fname, CHAR *aname, CHAR *embed, CHAR *delim) : CFeatures(fname)
{
    ASSERT(fname && aname && embed && delim);

    /* Allocate and generate gram configuration (words) */
    CIO::message(M_DEBUG, "Initializing Mindy gram features\n");    
    alph_type_t at = alph_get_type(aname);
    cfg = gram_cfg_words(alph_create(at), delim);
    set_embedding(cfg, embed);

    CIO::message(M_INFO, "Mindy in word mode (d: '%s', a: %s, e: %s)\n", 
                 delim, alph_get_name(at), gram_cfg_get_embed(cfg->embed));

    load(fname);
}

/**
 * Constructor for word features extracted from string features
 * @param sf String features to use
 * @param aname Alphabet name, e.g. bytes, ascii, text, dna
 * @param delim Escaped string of delimiters, e.g. '%20.,'
 * @param len   Length of byte array
 */
CMindyGramFeatures::CMindyGramFeatures(CStringFeatures<CHAR> *sf, CHAR *aname, CHAR *embed, CHAR *delim) : CFeatures(0)
{
    ASSERT(sf && aname && embed && delim);

    /* Allocate and generate gram configuration (words) */
    CIO::message(M_DEBUG, "Initializing Mindy gram features\n");    
    alph_type_t at = alph_get_type(aname);
    cfg = gram_cfg_words(alph_create(at), delim);
    set_embedding(cfg, embed);

    CIO::message(M_INFO, "Mindy in word mode (d: '%s', a: %s, e: %s)\n", 
                 delim, alph_get_name(at), gram_cfg_get_embed(cfg->embed));

    import(sf);
}

/**
 * Copy constructor for gram features
 * @param orig Gram feature object to copy
 */
CMindyGramFeatures::CMindyGramFeatures(const CMindyGramFeatures & orig) : CFeatures(orig)
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
 * Destructor for gram features
 */
CMindyGramFeatures::~CMindyGramFeatures()
{
    CIO::message(M_DEBUG, "Destroying Mindy gram features\n");
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
CFeatures *CMindyGramFeatures::duplicate() const
{
    return new CMindyGramFeatures(*this);
}

/**
 * Get gram vector for sample i
 * @param i index of gram vector
 * @return gram vector
 */
gram_t *CMindyGramFeatures::get_feature_vector(INT i)
{
    ASSERT(vectors != NULL);
    ASSERT(i >= 0 && i < num_vectors);

    return vectors[i];
}

/**
 * Set gram vector for sample i
 * @param num index of feature vector
 */
void CMindyGramFeatures::set_feature_vector(INT i, gram_t * g)
{
    ASSERT(vectors != NULL);
    ASSERT(i >= 0 && i < num_vectors);

    /* Destroy previous gram */
    if (vectors[i])
        gram_destroy(vectors[i]);

    vectors[i] = g;
}

void CMindyGramFeatures::set_embedding(gram_cfg_t *cfg, CHAR *embed)
{
    if (!strcasecmp(embed, "count"))
        gram_cfg_set_embed(cfg, GE_COUNT);
    else if (!strcasecmp(embed, "freq"))
        gram_cfg_set_embed(cfg, GE_FREQ);
    else if (!strcasecmp(embed, "bin")) 
        gram_cfg_set_embed(cfg, GE_BIN);
    else
        CIO::message(M_ERROR, "Unknown embedding '%s'\n", embed);    
}

/**
 * Get a feature (gram) from a gram vector
 * @param i Index of gram vector
 * @param j Index of feature in gram vector
 * @param b Buffer to hold gram of at least 65 bytes
 * @return gram (e.g. an n-gram or word)
 */
inline ULONG CMindyGramFeatures::get_feature(INT i, INT j)
{
    ASSERT(vectors && i < num_vectors);
    ASSERT(j < (signed) vectors[i]->num);

    return vectors[i]->gram[j];
}

/**
 * Get the length of gram vector at index i
 * @param i Index of gram vector
 * @return length of gram vector
 */
inline INT CMindyGramFeatures::get_vector_length(INT i)
{
    ASSERT(vectors && i < num_vectors);
    return vectors[i]->num;
}

/**
 * Imports gram features from a string feature object
 * @param sf String feature object
 * @return true on success, false otherwise
 */
bool CMindyGramFeatures::import(CStringFeatures < CHAR > *sf)
{
    INT i;
    num_vectors = sf->get_num_vectors();
    CIO::message(M_INFO, "Importing %ld string features\n", num_vectors);

    vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
    if (!vectors) {
        CIO::message(M_ERROR, "Could not allocate memory\n");
        return false;
    }

    for (i = 0; i < num_vectors; i++) {
        INT len;
        CHAR *s = sf->get_feature_vector(i, len);
        vectors[i] = gram_extract(cfg, (byte_t *) s, (size_t) len);

        CIO::message(M_DEBUG, "Extracted gram vector %d: %d grams\n", i, 
                     vectors[i]->num);
    }

    return true;
}

/**
 * Loads a set of strings and extracts corresponding gram vectors
 * @param fname File name
 * @return true on success, false otherwise
 */
bool CMindyGramFeatures::load(CHAR * fname)
{
    CIO::message(M_INFO, "Loading strings from %s\n", fname);
    LONG len = 0;
    CHAR *s, *t;

    CFile f(fname, 'r', F_CHAR);
    CHAR *data = f.load_char_data(NULL, len);

    if (!f.is_ok()) {
        CIO::message(M_ERROR, "Reading file failed\n");
        return false;
    }

    /* Count strings terminated by \n */
    num_vectors = 0;
    for (LONG i = 0; i < len; i++)
        if (data[i] == '\n')
            CIO::message(M_INFO, "File contains %ld string vectors\n",
                num_vectors);

    vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
    if (!vectors) {
        CIO::message(M_ERROR, "Could not allocate memory\n");
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
#endif
