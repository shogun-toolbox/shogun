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

			std::unique_ptr<RandomAccessFile> new_random_access_file(
				const std::string& fname) override
			{
				throw ShogunNotImplementedException("new_random_access_file unimplemented");
			}

			std::unique_ptr<WritableFile> new_writable_file(
				const std::string& fname) override
			{
				throw ShogunNotImplementedException("NewWritableFile new_writable_file");
			}

			std::unique_ptr<WritableFile> new_appendable_file(
				const std::string& fname) override
			{
				throw ShogunNotImplementedException("new_appendable_file unimplemented");
			}

			bool file_exists(const std::string& fname) override
			{
				throw ShogunNotImplementedException("file_exists unimplemented");
			}

			void delete_file(const std::string& fname) override
			{
				throw ShogunNotImplementedException("delete_file unimplemented");
			}

			void create_dir(const std::string& dirname) override
			{
				throw ShogunNotImplementedException("create_dir unimplemented");
			}

			void delete_dir(const std::string& dirname) override
			{
				throw ShogunNotImplementedException("delete_dir unimplemented");
			}

			uint64_t get_file_size(const std::string& fname) override
			{
				throw ShogunNotImplementedException("get_file_size unimplemented");
			}

			void rename_file(const std::string& src, const std::string& target) override
			{
				throw ShogunNotImplementedException("rename_file unimplemented");
			}

		};
	}
}

#endif

