#include <gtest/gtest.h>

#include <fstream>
#include <shogun/base/ShogunEnv.h>
#include <shogun/io/stream/FileOutputStream.h>

using namespace shogun;
using namespace std;

TEST(FileOutputStream, write)
{
	string fname = "test123";
	string_view test_str("asdf");
	auto fs_registry = env();
	auto r = fs_registry->file_exists(fname);
	ASSERT_TRUE(r);

	unique_ptr<io::WritableFile> file;
	r = fs_registry->new_writable_file(fname, &file);
	ASSERT_FALSE(r);
	auto fos = std::make_unique<io::FileOutputStream>(file.get());
	r = fos->write(test_str.data(), test_str.size());
	ASSERT_FALSE(r);
	r = fos->close();

	ifstream is(fname);
	char str_in[5];
	is.get(&str_in[0], 5);
	EXPECT_EQ(test_str, string(str_in));
	r = fs_registry->delete_file(fname);
	ASSERT_FALSE(r);
}
