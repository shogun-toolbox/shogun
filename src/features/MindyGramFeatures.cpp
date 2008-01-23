/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Konrad Rieck
 * Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
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
 * Destructor for gram features
 */
CMindyGramFeatures::~CMindyGramFeatures()
{
    SG_DEBUG( "Destroying Mindy gram features\n");
    /* Destroy gram vectors */
    for (INT i = 0; i < num_vectors; i++)
        gram_destroy(vectors[i]);
    free(vectors);

    /* Destroy configuration */
    micfg_destroy(cfg);
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

/**
 * Get a feature (gram) from a gram vector
 * @param i Index of gram vector
 * @param j Index of feature in gram vector
 * @param b Buffer to hold gram of at least 65 bytes
 * @return gram (e.g. an n-gram or word)
 */
ULONG CMindyGramFeatures::get_feature(INT i, INT j)
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
INT CMindyGramFeatures::get_vector_length(INT i)
{
    ASSERT(vectors && i < num_vectors);
    return vectors[i]->num;
}

/**
 * Trims the features to a maximum value
 * @param i Maximum value
 */
void CMindyGramFeatures::trim_max(double max)
{
    for (INT i = 0; i < num_vectors; i++)
        gram_trim_max(vectors[i], max);
} 

/**
 * Loads a set of strings and extracts corresponding gram vectors
 * @param fname File name
 * @return true on success, false otherwise
 */
bool CMindyGramFeatures::load(CHAR * fname)
{
    SG_INFO( "Loading strings from %s\n", fname);
    LONG len = 0;
    CHAR *s, *t;

    CFile f(fname, 'r', F_CHAR);
    CHAR *data = f.load_char_data(NULL, len);

    if (!f.is_ok()) {
        SG_ERROR( "Reading file failed\n");
        return false;
    }

    /* Count strings terminated by \n */
    num_vectors = 0;
    for (LONG i = 0; i < len; i++)
        if (data[i] == '\n')
            SG_INFO( "File contains %ld string vectors\n",
                num_vectors);

    vectors = (gram_t **) calloc(num_vectors, sizeof(gram_t *));
    if (!vectors) {
        SG_ERROR( "Could not allocate memory\n");
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
