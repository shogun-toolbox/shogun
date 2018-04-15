#ifdef _MSC_VER
#include <Shlwapi.h>
#include <Windows.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#undef StrCat
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <system_error>

#include <shogun/io/fs/WindowsFileSystem.h>

using namespace shogun;
using namespace std;

class WindowsRandomAccessFile: public RandomAccessFile
{
public:
	WindowsRandomAccessFile(const string& fname, HANDLE hfile):
		m_filename(fname), m_hfile(hfile) {}

	~WindowsRandomAccessFile() override
	{
		if (m_hfile != NULL && m_hfile != INVALID_HANDLE_VALUE)
			::CloseHandle(m_hfile);
	}

	void read(uint64_t offset, size_t n, Chunk* result, char* scratch) const override
	{
		char* dst = scratch;
		while (n > 0) {
			SSIZE_T r = pread(m_hfile, dst, n, offset);
			if (r > 0)
			{
				offset += r;
				dst += r;
				n -= r;
			}
			else if (r == 0)
			{
				throw system_error(
					make_error_code(errc::result_out_of_range),
					"Read less bytes than requested");

			}
			else if (errno == EINTR || errno == EAGAIN) {
				// Retry
			}
			else
			{
				throw system_error(::GetLastError(), system_category());
			}
			}
			*result = Chunk(scratch, dst - scratch);
	}

private:
	string m_filename;
	HANDLE m_hfile;
};

class WindowsWritableFile: public WritableFile
{
public:
	WindowsWritableFile(const string& fname, HANDLE hfile):
		m_filename(fname), m_hfile(hfile) {}

	~WindowsWritableFile() override
	{
		if (m_hfile != NULL && m_hfile != INVALID_HANDLE_VALUE)
      		WindowsWritableFile::close();
	}

	void append(const Chunk& data) override
	{
		DWORD bytes_written = 0;
		DWORD data_size = static_cast<DWORD>(data.size());
		BOOL write_result =
		    ::WriteFile(m_hfile, data.data(), data_size, &bytes_written, NULL);
		if (FALSE == write_result)
			throw system_error(::GetLastError(), system_category());

		assert(size_t(bytes_written) == data.size());
	}

	void close() override
	{
		assert(INVALID_HANDLE_VALUE != m_hfile);

		Status result = Flush();
		if (!result.ok()) {
		  return result;
		}

		if (FALSE == ::CloseHandle(m_hfile))
			throw system_error(::GetLastError(), system_category());

		m_hfile = INVALID_HANDLE_VALUE;
	}

	void flush() override
	{
		if (FALSE == ::FlushFileBuffers(m_hfile))
			throw system_error(::GetLastError(), system_category());
	}

	void sync() override
	{
		flush();
	}

private:
	string m_filename;
	HANDLE* m_hfile;
};

unique_ptr<RandomAccessFile> WindowsFileSystem::new_random_access_file(
 	const string& fname)
{
	string translated_fname = translate_name(fname);
	int fd = open(translated_fname.c_str(), O_RDONLY);
	if (fd < 0)
		throw system_error(errno, generic_category(), "could not create random access");
	return make_unique<PosixRandomAccessFile>(translated_fname, fd);
}

unique_ptr<WritableFile> WindowsFileSystem::new_writable_file(const string& fname)
{
	string translated_fname = translate_name(fname);
	FILE* f = fopen(translated_fname.c_str(), "w");
	if (f == nullptr)
		throw system_error(errno, generic_category(), "could not create writable file");
	return make_unique<PosixWritableFile>(translated_fname, f);
}

unique_ptr<WritableFile> WindowsFileSystem::new_appendable_file(const string& fname)
{
	string translated_fname = translate_name(fname);
	FILE* f = fopen(translated_fname.c_str(), "a");
	if (f == nullptr)
		throw system_error(errno, generic_category(), "could not create file");
	return make_unique<PosixWritableFile>(translated_fname, f);
}

bool WindowsFileSystem::file_exists(const string& fname)
{
	return (access(translate_name(fname).c_str(), F_OK) == 0)
		? true : false;
}

void WindowsFileSystem::delete_file(const string& fname)
{
	if (unlink(translate_name(fname).c_str()) != 0)
		throw system_error(errno, generic_category());
}

void WindowsFileSystem::create_dir(const string& name)
{
	if (mkdir(translate_name(name).c_str(), 0755) != 0)
		throw system_error(errno, generic_category());
}

void WindowsFileSystem::delete_dir(const string& name)
{
	if (rmdir(translate_name(name).c_str()) != 0)
		throw system_error(errno, generic_category());
}


void WindowsFileSystem::rename_file(const string& src, const string& target)
{
	if (rename(translate_name(src).c_str(), translate_name(target).c_str()) != 0)
		throw system_error(errno, generic_category());
}

REGISTER_FILE_SYSTEM("", WindowsFileSystem);
REGISTER_FILE_SYSTEM("file", LocalWindowsFileSystem);

#endif
