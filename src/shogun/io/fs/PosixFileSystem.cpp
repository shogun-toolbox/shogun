#ifndef _MSC_VER

#include <dirent.h>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#if !defined(__APPLE__)
#include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <system_error>

#include <shogun/io/fs/PosixFileSystem.h>

using namespace shogun;
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

	void read(uint64_t offset, size_t n, Chunk* result, char* scratch) const override
	{
		char* dst = scratch;
		while (n > 0)
		{
			ssize_t r = pread(m_fd, dst, n, static_cast<off_t>(offset));
			if (r > 0)
			{
				dst += r;
				n -= r;
				offset += r;
			}
			else if (r == 0)
			{
				throw system_error(make_error_code(errc::result_out_of_range), "Read less bytes than requested");
			}
			else if (errno == EINTR || errno == EAGAIN)
			{
				// should retry...
			}
			else
			{
				throw system_error(errno, generic_category());
			}
		}
		*result = Chunk(scratch, dst - scratch);
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

	void append(const void* data, size_t size) override
	{
		size_t r = fwrite(data, 1, size, m_file);
		if (r != size)
			throw system_error(errno, generic_category());
	}

	void close() override
	{
		if (fclose(m_file) != 0)
			throw system_error(errno, generic_category());
		m_file = nullptr;
	}

	void flush() override
	{
		if (fflush(m_file) != 0)
			throw system_error(errno, generic_category());
	}

	void sync() override
	{
		if (fflush(m_file) != 0)
			throw system_error(errno, generic_category());
	}

private:
	string m_filename;
	FILE* m_file;
};

unique_ptr<RandomAccessFile> PosixFileSystem::new_random_access_file(
 	const string& fname)
{
	string translated_fname = translate_name(fname);
	int fd = open(translated_fname.c_str(), O_RDONLY);
	if (fd < 0)
		throw system_error(errno, generic_category(), "could not create random access");
	return make_unique<PosixRandomAccessFile>(translated_fname, fd);
}

unique_ptr<WritableFile> PosixFileSystem::new_writable_file(const string& fname)
{
	string translated_fname = translate_name(fname);
	FILE* f = fopen(translated_fname.c_str(), "w");
	if (f == nullptr)
		throw system_error(errno, generic_category(), "could not create writable file");
	return make_unique<PosixWritableFile>(translated_fname, f);
}

unique_ptr<WritableFile> PosixFileSystem::new_appendable_file(const string& fname)
{
	string translated_fname = translate_name(fname);
	FILE* f = fopen(translated_fname.c_str(), "a");
	if (f == nullptr)
		throw system_error(errno, generic_category(), "could not create file");
	return make_unique<PosixWritableFile>(translated_fname, f);
}

bool PosixFileSystem::file_exists(const string& fname)
{
	return (access(translate_name(fname).c_str(), F_OK) == 0)
		? true : false;
}

void PosixFileSystem::delete_file(const string& fname)
{
	if (unlink(translate_name(fname).c_str()) != 0)
		throw system_error(errno, generic_category());
}

void PosixFileSystem::create_dir(const string& name)
{
	if (mkdir(translate_name(name).c_str(), 0755) != 0)
		throw system_error(errno, generic_category());
}

void PosixFileSystem::delete_dir(const string& name)
{
	if (rmdir(translate_name(name).c_str()) != 0)
		throw system_error(errno, generic_category());
}


void PosixFileSystem::rename_file(const string& src, const string& target)
{
	if (rename(translate_name(src).c_str(), translate_name(target).c_str()) != 0)
		throw system_error(errno, generic_category());
}

bool PosixFileSystem::is_directory(const std::string& fname)
{
	if (!file_exists(fname))
		return false;

	struct stat sbuf;
	if (stat(translate_name(fname).c_str(), &sbuf) != 0)
		return false;

	return S_ISDIR(sbuf.st_mode);
}

uint64_t PosixFileSystem::get_file_size(const std::string& fname)
{
	struct stat sbuf;
	if (stat(translate_name(fname).c_str(), &sbuf) != 0)
		throw system_error(errno, generic_category());

	return sbuf.st_size;
}

REGISTER_FILE_SYSTEM("", PosixFileSystem);
REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);

#endif
