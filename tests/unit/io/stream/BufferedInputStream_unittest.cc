/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <gtest/gtest.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>
#include <shogun/io/stream/BufferedInputStream.h>

using namespace std;
using namespace shogun;

static std::vector<int> kBufferSizes = {
	1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
	12, 13, 14, 15, 16, 17, 18, 19, 20, 65536
};

static error_condition write_to_file(io::FileSystemRegistry* fs,
	const string& fname, const string& content)
{
	std::unique_ptr<io::WritableFile> file;
	auto ec = fs->new_writable_file(fname, &file);
	if (ec)
		return ec;

	ec = file->append(content);
	if (!ec)
		ec = file->close();
	return ec;
}

TEST(BufferedInputStream, read)
{
	auto fs = io::FileSystemRegistry::instance();
	string filename = "bis_test";
	ASSERT_FALSE(write_to_file(fs, filename, "foobarbaz"));
	std::unique_ptr<io::RandomAccessFile> file;
	ASSERT_FALSE(fs->new_random_access_file(filename, &file));

	for (auto buf_size : kBufferSizes)
	{
		auto fis = std::make_unique<io::FileInputStream>(file.get());
		auto in = std::make_unique<io::BufferedInputStream>(fis.get(), buf_size);
		string read;
		EXPECT_EQ(0, in->tell());
		ASSERT_FALSE(in->read(&read, 3));
		EXPECT_EQ(read, "foo");
		EXPECT_EQ(3, in->tell());
		ASSERT_FALSE(in->read(&read, 0));
		EXPECT_EQ(read, "");
		EXPECT_EQ(3, in->tell());
		ASSERT_FALSE(in->read(&read, 4));
		EXPECT_EQ(read, "barb");
		EXPECT_EQ(7, in->tell());
		ASSERT_FALSE(in->read(&read, 0));
		EXPECT_EQ(read, "");
		EXPECT_EQ(7, in->tell());
		EXPECT_TRUE(io::is_out_of_range(in->read(&read, 5)));
		EXPECT_EQ(read, "az");
		EXPECT_EQ(9, in->tell());
		EXPECT_TRUE(io::is_out_of_range(in->read(&read, 5)));
		EXPECT_EQ(read, "");
		EXPECT_EQ(9, in->tell());
		ASSERT_FALSE(in->read(&read, 0));
		EXPECT_EQ(read, "");
		EXPECT_EQ(9, in->tell());
	}
	ASSERT_FALSE(fs->delete_file(filename));
}

TEST(BufferedInputStream, skip)
{
	auto fs = io::FileSystemRegistry::instance();
	string filename = "bis_test_skip";
	ASSERT_FALSE(write_to_file(fs, filename, "foobarbaz"));
	std::unique_ptr<io::RandomAccessFile> file;
	ASSERT_FALSE(fs->new_random_access_file(filename, &file));

	for (auto buf_size : kBufferSizes)
	{
		auto fis = std::make_unique<io::FileInputStream>(file.get());
		auto in = std::make_unique<io::BufferedInputStream>(fis.get(), buf_size);
		string read;
		EXPECT_EQ(0, in->tell());
		ASSERT_FALSE(in->skip(3));
		EXPECT_EQ(3, in->tell());
		ASSERT_FALSE(in->skip(0));
		EXPECT_EQ(3, in->tell());
		ASSERT_FALSE(in->read(&read, 2));
		EXPECT_EQ(read, "ba");
		EXPECT_EQ(5, in->tell());
		ASSERT_FALSE(in->skip(0));
		EXPECT_EQ(5, in->tell());
		ASSERT_FALSE(in->skip(2));
		EXPECT_EQ(7, in->tell());
		ASSERT_FALSE(in->read(&read, 1));
		EXPECT_EQ(read, "a");
		EXPECT_EQ(8, in->tell());
		EXPECT_TRUE(io::is_out_of_range(in->skip(5)));
		EXPECT_EQ(9, in->tell());
		EXPECT_TRUE(io::is_out_of_range(in->skip(5)));
		EXPECT_EQ(9, in->tell());
		EXPECT_TRUE(io::is_out_of_range(in->read(&read, 5)));
		EXPECT_EQ(read, "");
		EXPECT_EQ(9, in->tell());
	}
	ASSERT_FALSE(fs->delete_file(filename));
}
