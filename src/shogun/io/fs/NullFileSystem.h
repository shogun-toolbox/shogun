#ifndef __NULL_FILE_SYSTEM_H__
#define __NULL_FILE_SYSTEM_H__

#include <shogun/io/fs/FileSystem.h>
#include <shogun/lib/exception/ShogunNotImplementedException.h>

namespace shogun
{
	namespace io
	{
		class NullFileSystem : public FileSystem
		{
		public:
			NullFileSystem() {}

			~NullFileSystem() override = default;

			std::error_condition new_random_access_file(
				const std::string& fname, std::unique_ptr<RandomAccessFile>*) const override
			{
				throw ShogunNotImplementedException("new_random_access_file unimplemented");
			}

			std::error_condition new_writable_file(
				const std::string& fname, std::unique_ptr<WritableFile>*) const override
			{
				throw ShogunNotImplementedException("NewWritableFile new_writable_file");
			}

			std::error_condition new_appendable_file(
				const std::string& fname, std::unique_ptr<WritableFile>*) const override
			{
				throw ShogunNotImplementedException("new_appendable_file unimplemented");
			}

			std::error_condition file_exists(const std::string& fname) const override
			{
				throw ShogunNotImplementedException("file_exists unimplemented");
			}

			std::error_condition delete_file(const std::string& fname) const override
			{
				throw ShogunNotImplementedException("delete_file unimplemented");
			}

			std::error_condition create_dir(const std::string& dirname) const override
			{
				throw ShogunNotImplementedException("create_dir unimplemented");
			}

			std::error_condition delete_dir(const std::string& dirname) const override
			{
				throw ShogunNotImplementedException("delete_dir unimplemented");
			}

			int64_t get_file_size(const std::string& fname) const override
			{
				throw ShogunNotImplementedException("get_file_size unimplemented");
			}

			std::error_condition rename_file(const std::string& src, const std::string& target) const override
			{
				throw ShogunNotImplementedException("rename_file unimplemented");
			}

		};
	}
}

#endif

