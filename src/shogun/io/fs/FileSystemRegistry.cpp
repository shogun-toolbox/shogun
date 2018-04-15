#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/fs/FileSystemRegistry.h>
#include <shogun/lib/exception/ShogunException.h>

using namespace shogun;
using namespace std;

void FileSystemRegistry::register_fs(
	const string& scheme,
	FileSystemRegistry::Factory factory)
{
	lock_guard<mutex> lock(m_mutex);
	if (!m_fs_registry.emplace(
		scheme,	unique_ptr<FileSystem>(factory())).second)
	{
		throw ShogunException("File factor for " + scheme + " already registered");
	}
}

FileSystem* FileSystemRegistry::lookup(const string& scheme) const
{
	lock_guard<mutex> lock(m_mutex);
	const auto found = m_fs_registry.find(scheme);
	return (found == m_fs_registry.end())
		? nullptr
		: found->second.get();
}

FileSystem* FileSystemRegistry::get_file_system_for_file(
	const string& fname) const
{
	string scheme, host, path;
	//io::ParseURI(fname, &scheme, &host, &path);
	FileSystem* fs = lookup(scheme);
	if (!fs)
	{
		if (scheme.empty())
		  scheme = "[local]";
		throw ShogunException(
			"File system scheme '" + scheme +
			"' not implemented (file: '" + scheme + "')");
	}
	return fs;
}


vector<string> FileSystemRegistry::get_registered_file_system_schemes()
{
	vector<string> schemes;
  	lock_guard<mutex> lock(m_mutex);
  	for (const auto& e: m_fs_registry)
    	schemes.push_back(e.first);
    return schemes;
}

unique_ptr<RandomAccessFile> FileSystemRegistry::new_random_access_file(const string& fname)
{
	return get_file_system_for_file(fname)->new_random_access_file(fname);
}


unique_ptr<WritableFile> FileSystemRegistry::new_writable_file(const string& fname)
{
	return get_file_system_for_file(fname)->new_writable_file(fname);
}

unique_ptr<WritableFile> FileSystemRegistry::new_appendable_file(const string& fname)
{
	return get_file_system_for_file(fname)->new_appendable_file(fname);
}

bool FileSystemRegistry::file_exists(const string& fname)
{
	return get_file_system_for_file(fname)->file_exists(fname);
}

void FileSystemRegistry::delete_file(const string& fname)
{
	return get_file_system_for_file(fname)->delete_file(fname);
}

void FileSystemRegistry::create_dir(const string& dirname)
{
	return get_file_system_for_file(dirname)->create_dir(dirname);
}

void FileSystemRegistry::delete_dir(const string& dirname)
{
	return get_file_system_for_file(dirname)->delete_dir(dirname);
}

void FileSystemRegistry::rename_file(const string& src, const string& target)
{
	FileSystem* src_fs = get_file_system_for_file(src);
	FileSystem* target_fs = get_file_system_for_file(src);
	if (src_fs != target_fs)
		throw ShogunException(
			"Renaming "+ src +" to "+ target +" not implemented");

	return src_fs->rename_file(src, target);
}

string FileSystemRegistry::translate_name(const string& name) const
{
	return get_file_system_for_file(name)->translate_name(name);
}

bool FileSystemRegistry::is_directory(const string& fname)
{
	return get_file_system_for_file(fname)->is_directory(fname);
}

uint64_t FileSystemRegistry::get_file_size(const string& fname)
{
	return get_file_system_for_file(fname)->get_file_size(fname);
}

