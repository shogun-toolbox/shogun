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
	class ShogunEnv;

	namespace io
	{
		class FileSystem;
		class RandomAccessFile;
		class WritableFile;

		class FileSystemRegistry
		{
			friend class shogun::ShogunEnv;

		public:
			typedef std::function<FileSystem*()> Factory;

			/**
			 *
			 */
			void register_fs(const std::string& scheme, const Factory& factory);

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
			std::vector<std::string> get_registered_file_system_schemes() const;

			std::error_condition new_random_access_file(const std::string& fname, std::unique_ptr<RandomAccessFile>*) const;

			std::error_condition new_writable_file(const std::string& fname, std::unique_ptr<WritableFile>*) const;

			std::error_condition new_appendable_file(const std::string& fname, std::unique_ptr<WritableFile>*) const;

			std::error_condition file_exists(const std::string& fname) const;

			std::error_condition delete_file(const std::string& fname) const;

			std::error_condition create_dir(const std::string& dirname) const;

			std::error_condition delete_dir(const std::string& dirname) const;

			std::error_condition rename_file(const std::string& src, const std::string& target) const;

			std::string translate_name(const std::string& name) const;
			std::error_condition is_directory(const std::string& fname) const;

			std::error_condition get_children(const std::string& dir,
				std::vector<std::string>* result) const;

			std::error_condition get_paths(const std::string& pattern,
				std::vector<std::string>* results) const;

			int64_t get_file_size(const std::string& fname) const;

		private:
			FileSystemRegistry() {}

			mutable std::mutex m_mutex;
			std::unordered_map<std::string, std::unique_ptr<FileSystem>> m_fs_registry;
		};
	}
}

#endif /* SHOGUN_FILESYSTEM_REGISTRY_H */
