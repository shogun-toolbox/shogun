/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2017 Leon Kuchenbecker
 */

#include <shogun/features/Alphabet.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(AlphabetTest, test_clone)
{
    CAlphabet * alph = new CAlphabet(PROTEIN);
    CAlphabet * alph_clone = (CAlphabet *) alph->clone();

    EXPECT_EQ(alph->get_num_symbols(), alph_clone->get_num_symbols());
    EXPECT_EQ(alph->get_num_symbols_in_histogram(), alph_clone->get_num_symbols_in_histogram());

    SG_UNREF(alph);
    SG_UNREF(alph_clone);
}
