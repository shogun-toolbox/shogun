/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef SHOGUN_FILESYSTEM_REGISTRY_H
#define SHOGUN_FILESYSTEM_REGISTRY_H

#include <mutex>
#include <unordered_map>

namespace shogun
{

class FileSystem;
class RandomAccessFile;
class WritableFile;

class FileSystemRegistry
{
public:
	typedef std::function<FileSystem*()> Factory;


	static FileSystemRegistry* instance() {
		static FileSystemRegistry* fsr = new FileSystemRegistry();
		return fsr;
	}

	/**
	 *
	 */
	void register_fs(const std::string& scheme, Factory factory);

	/**
	 *
	 */
	FileSystem* lookup(const std::string& scheme) const;


	/**
	 *
	 */
	FileSystem* get_file_system_for_file(const std::string& fname) const;

	/**
	 *
	 */
	std::vector<std::string> get_registered_file_system_schemes();

	std::unique_ptr<RandomAccessFile> new_random_access_file(const std::string& fname);

	std::unique_ptr<WritableFile> new_writable_file(const std::string& fname);

	std::unique_ptr<WritableFile> new_appendable_file(const std::string& fname);

	bool file_exists(const std::string& fname);

	void delete_file(const std::string& fname);

	void create_dir(const std::string& dirname);

	void delete_dir(const std::string& dirname);

	void rename_file(const std::string& src, const std::string& target);

	std::string translate_name(const std::string& name) const;
	bool is_directory(const std::string& fname);

	uint64_t get_file_size(const std::string& fname);

private:
	FileSystemRegistry() {}

	mutable std::mutex m_mutex;
  	std::unordered_map<std::string, std::unique_ptr<FileSystem>> m_fs_registry;
};

}

#endif /* SHOGUN_FILESYSTEM_REGISTRY_H */
