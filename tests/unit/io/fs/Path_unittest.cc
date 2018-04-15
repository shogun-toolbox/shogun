/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <gtest/gtest.h>
#include <shogun/io/fs/Path.h>

using namespace shogun::io;

TEST(PathTest, join_path)
{
	EXPECT_EQ("/foo/bar", join_path("/foo", "bar"));
	EXPECT_EQ("foo/bar", join_path("foo", "bar"));
	EXPECT_EQ("foo/bar", join_path("foo", "/bar"));
	EXPECT_EQ("/foo/bar", join_path("/foo", "/bar"));

	EXPECT_EQ("/bar", join_path("", "/bar"));
	EXPECT_EQ("bar", join_path("", "bar"));
	EXPECT_EQ("/foo", join_path("/foo", ""));
}

TEST(PathTest, is_absolute_path)
{
	EXPECT_FALSE(is_absolute_path(""));
	EXPECT_FALSE(is_absolute_path("../foo"));
	EXPECT_FALSE(is_absolute_path("foo"));
	EXPECT_FALSE(is_absolute_path("./foo"));
	EXPECT_FALSE(is_absolute_path("foo/bar/baz/"));
	EXPECT_TRUE(is_absolute_path("/foo"));
	EXPECT_TRUE(is_absolute_path("/foo/bar/../baz"));
}

#define EXPECT_URI(uri, scheme, host, path) \
	do { \
		std::string_view u(uri); \
		std::string_view s, h, p; \
		parse_uri(u, &s, &h, &p); \
		EXPECT_EQ(scheme, s); \
		EXPECT_EQ(host, h); \
		EXPECT_EQ(path, p); \
	} while(0);

TEST(PathTest, parse_uri)
{
	EXPECT_URI("/usr/local/bin", "", "", "/usr/local/bin");
	EXPECT_URI("file:///usr/local/bin", "file", "", "/usr/local/bin");
	EXPECT_URI("hdfs://localhost:8020/path/to/file",
		"hdfs", "localhost:8020", "/path/to/file");
}

#undef EXPECT_URI
