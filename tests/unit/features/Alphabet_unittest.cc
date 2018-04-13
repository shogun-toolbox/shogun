/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Leon Kuchenbecker
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
