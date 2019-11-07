#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/fs/FileSystemRegistry.h>
#include <shogun/io/fs/Path.h>
#include <shogun/lib/exception/ShogunException.h>

#include <sstream>

using namespace shogun::io;
using namespace std;

void FileSystemRegistry::register_fs(
	const string& scheme,
	const FileSystemRegistry::Factory& factory)
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
	string_view scheme, host, path;
	parse_uri(fname, &scheme, &host, &path);
	auto fs = lookup(string(scheme));
	if (!fs)
	{
		if (scheme.empty())
			scheme = "[local]";
		stringstream ss;
		ss << "File system scheme '" << scheme
			<< "' not implemented (file: '" << scheme << "')";
		throw ShogunException(ss.str());
	}
	return fs;
}


vector<string> FileSystemRegistry::get_registered_file_system_schemes() const
{
	vector<string> schemes;
  	lock_guard<mutex> lock(m_mutex);
  	for (const auto& e: m_fs_registry)
    	schemes.push_back(e.first);
    return schemes;
}

error_condition FileSystemRegistry::new_random_access_file(const string& fname, unique_ptr<RandomAccessFile>* file) const
{
	return get_file_system_for_file(fname)->new_random_access_file(fname, file);
}


error_condition FileSystemRegistry::new_writable_file(const string& fname, unique_ptr<WritableFile>* file) const
{
	return get_file_system_for_file(fname)->new_writable_file(fname, file);
}

error_condition FileSystemRegistry::new_appendable_file(const string& fname, unique_ptr<WritableFile>* file) const
{
	return get_file_system_for_file(fname)->new_appendable_file(fname, file);
}

error_condition FileSystemRegistry::file_exists(const string& fname) const
{
	return get_file_system_for_file(fname)->file_exists(fname);
}

error_condition FileSystemRegistry::delete_file(const string& fname) const
{
	return get_file_system_for_file(fname)->delete_file(fname);
}

error_condition FileSystemRegistry::create_dir(const string& dirname) const
{
	return get_file_system_for_file(dirname)->create_dir(dirname);
}

error_condition FileSystemRegistry::delete_dir(const string& dirname) const
{
	return get_file_system_for_file(dirname)->delete_dir(dirname);
}

error_condition FileSystemRegistry::rename_file(const string& src, const string& target) const
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

error_condition FileSystemRegistry::is_directory(const string& fname) const
{
	return get_file_system_for_file(fname)->is_directory(fname);
}

int64_t FileSystemRegistry::get_file_size(const string& fname) const
{
	return get_file_system_for_file(fname)->get_file_size(fname);
}


error_condition FileSystemRegistry::get_children(const string& dir,
	vector<string>* result) const
{
	return get_file_system_for_file(dir)->get_children(dir, result);
}

error_condition FileSystemRegistry::get_paths(const string& pattern,
	vector<string>* results) const
{
	return get_file_system_for_file(pattern)->get_paths(pattern, results);
}
