/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Devramx
 */
#include<gtest/gtest.h>
#include <shogun/distance/LevensteinDistance.h>
using namespace shogun;

TEST(LevensteinDistance,Basecase)
{
	EXPECT_EQ(Levenstein("",""),0);
	EXPECT_EQ(Levenstein("a","b"),2);
	EXPECT_EQ(Levenstein("a","a"),0);
	EXPECT_EQ(Levenstein("abcd",""),4);
	EXPECT_EQ(Levenstein("","abcde"),5);
	EXPECT_EQ(Levenstein("abc","axy"),4);
	EXPECT_EQ(Levenstein("abcd","abed"),2);
}

