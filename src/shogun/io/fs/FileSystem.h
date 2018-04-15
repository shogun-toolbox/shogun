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
#include <vector>

#include <shogun/base/macros.h>
#include <shogun/lib/chunk.h>
#include <shogun/io/fs/FileSystemRegistry.h>

namespace shogun
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
	virtual std::unique_ptr<RandomAccessFile> new_random_access_file(const std::string& fname) = 0;

	/**
	 * Create a new writable file
	 *
     * @param name file name string
     * @return unique pointer to the file or exception in case of error
     */
	virtual std::unique_ptr<WritableFile> new_writable_file(const std::string& fname) = 0;

	/**
	 * Create a new writable file starting on the end of the file
	 *
     * @param name file name string
     * @return unique pointer to the file or exception in case of error
     */
	virtual std::unique_ptr<WritableFile> new_appendable_file(const std::string& fname) = 0;

	/**
	 * Check if file exists
	 *
	 * @return True in case the file exists False otherwise.
	 */
	virtual bool file_exists(const std::string& fname) = 0;

	/**
	 * Delete a given file
	 *
	 * @param fname file name to be deleted
	 */
	virtual void delete_file(const std::string& fname) = 0;

	/**
	 * Create directory
	 *
	 * @param dirname name of the directory to create
	 */
	virtual void create_dir(const std::string& dirname) = 0;

	/**
	 * Delete directory
	 *
	 * @param dirname name of the directory to delete
	 */
	virtual void delete_dir(const std::string& dirname) = 0;

	/**
	 * Rename file
	 *
	 * @param src file name to rename
	 * @param target of the renaming
	 */
	virtual void rename_file(const std::string& src, const std::string& target) = 0;

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
	virtual bool is_directory(const std::string& fname) = 0;

	virtual uint64_t get_file_size(const std::string& fname) = 0;
};

/**
 * A file abstraction for randomly reading the contents of a file.
 */
class RandomAccessFile
{
public:
	RandomAccessFile() {}
	virtual ~RandomAccessFile() {}

	virtual void read(
		uint64_t offset, size_t n,
		Chunk* result, char* scratch) const = 0;
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
	virtual void append(const void* data, size_t size) = 0;

	/**
	 * Close the file
	 */
	virtual void close() = 0;

	/**
	 * Flush the file
	 */
	virtual void flush() = 0;

	/**
	 * Sync the content of the file to the file system
	 */
	virtual void sync() = 0;

private:
	SG_DELETE_COPY_AND_ASSIGN(WritableFile);
};

namespace internal {

template <typename Factory>
struct FileSystemRegister
{
	FileSystemRegister(FileSystemRegistry* fsr, const std::string& scheme)
	{
		fsr->register_fs(scheme, []() -> FileSystem* { return new Factory; });
	}
};

} // namespace internal

#define REGISTER_FILE_SYSTEM_FACTORY(fsr, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, fsr, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, fsr, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, fsr, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, fsr, scheme, factory)   \
  static ::shogun::internal::FileSystemRegister<factory>       \
      register_ff##ctr SG_ATTRIBUTE_UNUSED =                   \
          ::shogun::internal::FileSystemRegister<factory>(fsr, scheme);

#define REGISTER_FILE_SYSTEM(scheme, factory) \
	REGISTER_FILE_SYSTEM_FACTORY(FileSystemRegistry::instance(), scheme, factory);

} // namespace shogun

#endif
