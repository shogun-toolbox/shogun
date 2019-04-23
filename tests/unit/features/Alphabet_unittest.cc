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
    auto alph = std::make_shared<Alphabet>(PROTEIN);
    auto alph_clone = alph->clone()->as<Alphabet>();

    EXPECT_EQ(alph->get_num_symbols(), alph_clone->get_num_symbols());
    EXPECT_EQ(alph->get_num_symbols_in_histogram(), alph_clone->get_num_symbols_in_histogram());



}
