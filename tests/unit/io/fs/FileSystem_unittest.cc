/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <gtest/gtest.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/fs/FileSystemRegistry.h>
#include <shogun/io/fs/Path.h>

using namespace std;
using namespace shogun;

class FileSystem : public ::testing::Test
{
public:
	io::FileSystemRegistry* fs_registry;
	virtual void SetUp()
	{
		fs_registry = env();
	}

	virtual void TearDown() {}
};

TEST_F(FileSystem, file_operations)
{
	// file name current unix timestamp should not exist
	auto unix_timestamp_str = std::to_string(std::chrono::seconds(std::time(NULL)).count());
	auto r = fs_registry->file_exists(unix_timestamp_str);
	ASSERT_TRUE(r);

	// create file
	std::unique_ptr<io::WritableFile> file;
	r = fs_registry->new_writable_file(unix_timestamp_str, &file);
	ASSERT_FALSE(r);
	file->close();

	// now it should already exist
	r = fs_registry->file_exists(unix_timestamp_str);
	ASSERT_FALSE(r);

	// file should be empty
	auto size = fs_registry->get_file_size(unix_timestamp_str);
	ASSERT_EQ(0L, size);

	r = fs_registry->new_appendable_file(unix_timestamp_str, &file);
	ASSERT_FALSE(r);
	string_view content("test");
	file->append(content);
	file->close();

	size = fs_registry->get_file_size(unix_timestamp_str);
	ASSERT_EQ(content.size(), size);

	std::unique_ptr<io::RandomAccessFile> file_read;
	r = fs_registry->new_random_access_file(unix_timestamp_str, &file_read);
	ASSERT_FALSE(r);
	string buffer;
	buffer.resize(4);
	string_view result;
	r = file_read->read(0, 4, &result, &(buffer[0]));
	file->close();
	ASSERT_EQ(content, result);

	// delete file
	r = fs_registry->delete_file(unix_timestamp_str);
	ASSERT_FALSE(r);

	// back to square one
	r = fs_registry->file_exists(unix_timestamp_str);
	ASSERT_TRUE(r);
}

TEST_F(FileSystem, rename_file)
{
	auto from = std::to_string(std::chrono::milliseconds(std::time(NULL)).count());
	auto to = std::to_string(std::chrono::milliseconds(std::time(NULL)).count()+1);

	auto r = fs_registry->file_exists(from);
	ASSERT_TRUE(r);
	r = fs_registry->file_exists(to);
	ASSERT_TRUE(r);

	std::unique_ptr<io::WritableFile> file;
	r = fs_registry->new_writable_file(from, &file);
	ASSERT_FALSE(r);
	file->close();

	r = fs_registry->rename_file(from, to);
	ASSERT_FALSE(r);

	r = fs_registry->file_exists(to);
	ASSERT_FALSE(r);

	r = fs_registry->delete_file(to);
	ASSERT_FALSE(r);
}

TEST_F(FileSystem, dir_operations)
{
	auto r = fs_registry->is_directory(".");
	ASSERT_FALSE(r);

	auto dir = std::to_string(std::chrono::milliseconds(std::time(NULL)).count());
	r = fs_registry->file_exists(dir);
	ASSERT_TRUE(r);

	r = fs_registry->is_directory(dir);
	ASSERT_TRUE(r);

	r = fs_registry->create_dir(dir);
	ASSERT_FALSE(r);

	r = fs_registry->is_directory(dir);
	ASSERT_FALSE(r);

	r = fs_registry->create_dir(io::join_path(dir, dir));
	ASSERT_FALSE(r);

	vector<string> children;
	r = fs_registry->get_children(dir, &children);
	ASSERT_FALSE(r);
	ASSERT_EQ(1, children.size());

	r = fs_registry->delete_dir(io::join_path(dir, dir));
	ASSERT_FALSE(r);

	r = fs_registry->delete_dir(dir);
	ASSERT_FALSE(r);

	r = fs_registry->is_directory(dir);
	ASSERT_TRUE(r);
}
/*


TEST_F(FileSystem, get_paths)
{
//			get_paths(const std::string& pattern,
//				std::vector<std::string>* results) const = 0;

}
*/
