#include <gtest/gtest.h>

#include <fstream>
#include <shogun/base/some.h>
#include <shogun/io/stream/FileOutputStream.h>

using namespace shogun;
using namespace std;

TEST(FileOutputStream, write)
{
	string fname = "test123";
	std::string test_str("asdf");
	auto fs_registry = FileSystemRegistry::instance();
	auto file = fs_registry->new_writable_file(fname);
	auto fos = Some<CFileOutputStream>::from_raw(new CFileOutputStream(std::move(file)));
	fos->write(test_str.data(), test_str.length());
	fos->close();

	std::ifstream is(fname);
	char str_in[5];
	is.get(&str_in[0], 5);
	EXPECT_EQ(test_str, std::string(str_in));
	unlink(fname.c_str());
}
