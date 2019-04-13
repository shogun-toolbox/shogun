/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef SHOGUN_FILESYSTEM_H
#define SHOGUN_FILESYSTEM_H

#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <shogun/base/macros.h>
#include <shogun/io/fs/FileSystemRegistry.h>
#include <string_view>

namespace shogun
{
	namespace io
	{
		class RandomAccessFile;
		class WritableFile;

		/**
		 * Interface representing a filesystem.
		 * A unified interface to open read and write files.
		 */
		class FileSystem
		{
		public:
			FileSystem() {}

			virtual ~FileSystem() {}

			/**
			 * Create a new random access read-only file
			 *
			 * @param name file name string
			 * @return unique pointer to the file or exception in case of error
			 */
			virtual std::error_condition new_random_access_file(const std::string& fname, std::unique_ptr<RandomAccessFile>*) const = 0;

			/**
			 * Create a new writable file
			 *
			 * @param name file name string
			 * @return unique pointer to the file or exception in case of error
			 */
			virtual std::error_condition new_writable_file(const std::string& fname, std::unique_ptr<WritableFile>*) const = 0;

			/**
			 * Create a new writable file starting on the end of the file
			 *
			 * @param name file name string
			 * @return unique pointer to the file or exception in case of error
			 */
			virtual std::error_condition new_appendable_file(const std::string& fname, std::unique_ptr<WritableFile>*) const = 0;

			/**
			 * Check if file exists
			 *
			 * @return True in case the file exists False otherwise.
			 */
			virtual std::error_condition file_exists(const std::string& fname) const = 0;

			/**
			 * Delete a given file
			 *
			 * @param fname file name to be deleted
			 */
			virtual std::error_condition delete_file(const std::string& fname) const = 0;

			/**
			 * Create directory
			 *
			 * @param dirname name of the directory to create
			 */
			virtual std::error_condition create_dir(const std::string& dirname) const = 0;

			/**
			 * Delete directory
			 *
			 * @param dirname name of the directory to delete
			 */
			virtual std::error_condition delete_dir(const std::string& dirname) const = 0;

			/**
			 * Rename file
			 *
			 * @param src file name to rename
			 * @param target of the renaming
			 */
			virtual std::error_condition rename_file(const std::string& src, const std::string& target) const = 0;

			/**
			 * Translate name.
			 * Resolves and cleans up the URI
			 *
			 * @param name the URI of the file
			 */
			virtual std::string translate_name(const std::string& name) const
			{
				// TODO: clean the path from junk
				return name;
			}

			/**
			 * Checks whether a given path is a directory or not.
			 *
			 * @param name path to the directory
			 */
			virtual std::error_condition is_directory(const std::string& fname) const = 0;

			virtual std::error_condition get_children(const std::string& dir,
				std::vector<std::string>* result) const = 0;

			virtual std::error_condition get_paths(const std::string& pattern,
				std::vector<std::string>* results) const = 0;

			virtual int64_t get_file_size(const std::string& fname) const = 0;
		};

		/**
		 * A file abstraction for randomly reading the contents of a file.
		 */
		class RandomAccessFile
		{
		public:
			RandomAccessFile() {}
			virtual ~RandomAccessFile() {}

			virtual std::error_condition read(
				uint64_t offset, size_t n,
				std::string_view* result,
				char* scratch) const = 0;
		private:
			SG_DELETE_COPY_AND_ASSIGN(RandomAccessFile);
		};

		/**
		 * A file abstraction for sequentially writing out a file.
		 */
		class WritableFile
		{
		public:
			WritableFile() {}
			virtual ~WritableFile() {}

			/**
			 * append data to file
			 */
			virtual std::error_condition append(std::string_view data) = 0;

			/**
			 * Close the file
			 */
			virtual std::error_condition close() = 0;

			/**
			 * Flush the file
			 */
			virtual std::error_condition flush() = 0;

			/**
			 * Sync the content of the file to the file system
			 */
			virtual std::error_condition sync() = 0;

		private:
			SG_DELETE_COPY_AND_ASSIGN(WritableFile);
		};

		namespace detail
		{
			template <typename Factory>
			struct FileSystemRegister
			{
				FileSystemRegister(FileSystemRegistry* fsr, const std::string& scheme)
				{
					fsr->register_fs(scheme, []() -> FileSystem* { return new Factory; });
				}
			};
		} // namespace detail

#define REGISTER_FILE_SYSTEM_FACTORY(fsr, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, fsr, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, fsr, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, fsr, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, fsr, scheme, factory)   \
  static ::shogun::io::detail::FileSystemRegister<factory>       \
	  register_ff##ctr SG_ATTRIBUTE_UNUSED =                   \
		  ::shogun::io::detail::FileSystemRegister<factory>(fsr, scheme);

#define REGISTER_FILE_SYSTEM(scheme, factory) \
	REGISTER_FILE_SYSTEM_FACTORY(FileSystemRegistry::instance(), scheme, factory);

	} // namespace io
} // namespace shogun

#endif
