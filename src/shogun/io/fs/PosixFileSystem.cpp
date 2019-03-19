#ifndef _MSC_VER

#include <iostream>
#include <dirent.h>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <system_error>

#include <shogun/io/fs/PosixFileSystem.h>
#include <shogun/io/ShogunErrc.h>

using namespace shogun;
using namespace shogun::io;
using namespace std;

class PosixRandomAccessFile: public RandomAccessFile
{
public:
	PosixRandomAccessFile(const string& fname, int fd):
		m_filename(fname),
		m_fd(fd)
	{
	}

	~PosixRandomAccessFile() override
	{
		close(m_fd);
	}

	error_condition read(uint64_t offset, size_t n, string_view* result, char* scratch) const override
	{
		char* dst = scratch;
		error_condition ec {};
		while (n > 0 && !ec)
		{
			auto r = pread(m_fd, dst, n, static_cast<off_t>(offset));
			if (r > 0)
			{
				dst += r;
				n -= r;
				offset += r;
			}
			else if (r == 0)
			{
				// reach the end of file
				ec = make_error_condition(ShogunErrc::OutOfRange);
			}
			else if (errno == EINTR || errno == EAGAIN)
			{
				// retry...
			}
			else
			{
				ec = generic_category().default_error_condition(errno);
			}
		}
		*result = string_view(scratch, dst - scratch);
		return ec;
	}

private:
  string m_filename;
  int m_fd;
};

class PosixWritableFile: public WritableFile
{
public:
	PosixWritableFile(const string& fname, FILE* f):
		m_filename(fname),
		m_file(f)
	{
	}

	~PosixWritableFile() override
	{
		if (m_file != nullptr)
		{
			// Ignoring any potential errors
			fclose(m_file);
		}
	}

	error_condition append(string_view data) override
	{
		auto r = fwrite(data.data(), 1, data.size(), m_file);
		if (r != data.size())
			return generic_category().default_error_condition(errno);
		return {};
	}

	error_condition close() override
	{
		if (m_file == nullptr)
			return generic_category().default_error_condition(EBADF);
		if (fclose(m_file) != 0)
			return generic_category().default_error_condition(errno);
		m_file = nullptr;
		return {};
	}

	error_condition flush() override
	{
		if (fflush(m_file) != 0)
			return generic_category().default_error_condition(errno);
		return {};
	}

	error_condition sync() override
	{
		if (fflush(m_file) != 0)
			return generic_category().default_error_condition(errno);
		return {};
	}

private:
	string m_filename;
	FILE* m_file;
};

error_condition PosixFileSystem::new_random_access_file(
 	const string& fname, unique_ptr<RandomAccessFile>* file) const
{
	string translated_fname = translate_name(fname);
	int fd = open(translated_fname.c_str(), O_RDONLY);
	if (fd < 0)
		return generic_category().default_error_condition(errno);

	file->reset(new PosixRandomAccessFile(translated_fname, fd));
	return {};
}

error_condition PosixFileSystem::new_writable_file(const string& fname, unique_ptr<WritableFile>* file) const
{
	string translated_fname = translate_name(fname);
	FILE* f = fopen(translated_fname.c_str(), "w");
	if (f == nullptr)
		return generic_category().default_error_condition(errno);
	file->reset(new PosixWritableFile(translated_fname, f));
	return {};
}

error_condition PosixFileSystem::new_appendable_file(const string& fname, unique_ptr<WritableFile>* file) const
{
	string translated_fname = translate_name(fname);
	FILE* f = fopen(translated_fname.c_str(), "a");
	if (f == nullptr)
		return generic_category().default_error_condition(errno);
	file->reset(new PosixWritableFile(translated_fname, f));
	return {};
}

error_condition PosixFileSystem::file_exists(const string& fname) const
{
	int r = access(translate_name(fname).c_str(), F_OK);
	if (r != 0)
		return generic_category().default_error_condition(errno);
	return {};
}

error_condition PosixFileSystem::delete_file(const string& fname) const
{
	if (unlink(translate_name(fname).c_str()) != 0)
		return generic_category().default_error_condition(errno);
	return {};
}

error_condition PosixFileSystem::create_dir(const string& name) const
{
	if (mkdir(translate_name(name).c_str(), 0755) != 0)
		return generic_category().default_error_condition(errno);
	return {};
}

error_condition PosixFileSystem::delete_dir(const string& name) const
{
	if (rmdir(translate_name(name).c_str()) != 0)
		return generic_category().default_error_condition(errno);
	return {};
}


error_condition PosixFileSystem::rename_file(const string& src, const string& target) const
{
	if (rename(translate_name(src).c_str(), translate_name(target).c_str()) != 0)
		return generic_category().default_error_condition(errno);
	return {};
}

error_condition PosixFileSystem::is_directory(const string& fname) const
{
	auto r = file_exists(fname);
	if (r)
		return r;

	struct stat sbuf;
	if (stat(translate_name(fname).c_str(), &sbuf) != 0)
		return generic_category().default_error_condition(errno);

	return {};
}

uint64_t PosixFileSystem::get_file_size(const string& fname) const
{
	struct stat sbuf;
	if (stat(translate_name(fname).c_str(), &sbuf) != 0)
		return -1;//generic_category().default_error_condition(errno);

	return sbuf.st_size;
}

error_condition PosixFileSystem::get_children(const string& dir,
	vector<string>* result) const
{
	auto translated_dir = translate_name(dir);
	result->clear();

	DIR* d = opendir(translated_dir.c_str());
	if (d == nullptr)
		return generic_category().default_error_condition(errno);

	struct dirent* entry;
	while ((entry = readdir(d)) != nullptr)
	{
		string_view basename = entry->d_name;
		if ((basename != ".") && (basename != ".."))
			result->push_back(entry->d_name);
	}
	closedir(d);
	return {};
}

error_condition PosixFileSystem::get_paths(const string& pattern,
	vector<string>* results) const
{

}

REGISTER_FILE_SYSTEM("", PosixFileSystem);
REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);

#endif
