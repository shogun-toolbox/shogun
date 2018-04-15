#ifndef __POSIX_FILE_SYSTEM_H__
#define __POSIX_FILE_SYSTEM_H__
#ifndef _MSC_VER

#include <shogun/io/fs/FileSystem.h>

namespace shogun
{

class PosixFileSystem : public FileSystem
{
public:
	PosixFileSystem() {}

	~PosixFileSystem() {}

	std::unique_ptr<RandomAccessFile> new_random_access_file(
		const std::string& filename) override;

	std::unique_ptr<WritableFile> new_writable_file(
		const std::string& fname) override;

	std::unique_ptr<WritableFile> new_appendable_file(
		const std::string& fname) override;

	bool file_exists(const std::string& fname) override;

	void delete_file(const std::string& fname) override;

	void create_dir(const std::string& name) override;

	void delete_dir(const std::string& name) override;

	void rename_file(const std::string& src, const std::string& target) override;

	uint64_t get_file_size(const std::string& fname) override;

	bool is_directory(const std::string& fname) override;
};

class LocalPosixFileSystem : public PosixFileSystem
{
public:
	std::string translate_name(const std::string& name) const override
	{
		Chunk scheme, host, path;
		return path.to_string();
	}
};

}  // namespace shogun

#endif
#endif  // __POSIX_FILE_SYSTEM_H__
