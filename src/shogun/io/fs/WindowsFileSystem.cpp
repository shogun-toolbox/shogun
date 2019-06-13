#ifdef _MSC_VER
#include <Shlwapi.h>
// remove min and max macros (see numeric_limits<T>::max())
#define NOMINMAX
#include <Windows.h>
#include <assert.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#undef StrCat
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <limits>
#include <system_error>

#include <shogun/io/fs/WindowsFileSystem.h>
#include <shogun/io/ShogunErrc.h>

using namespace shogun;
using namespace shogun::io;
using namespace std;

const auto CloseHandleFunc = [](HANDLE h) { ::CloseHandle(h); };
typedef unique_ptr<void, decltype(CloseHandleFunc)> UniqueCloseHandlePtr;

SG_FORCED_INLINE wstring utf8_to_wchar(const string& utf8str)
{
	int size_required = MultiByteToWideChar(
		CP_UTF8, 0, utf8str.c_str(),
		(int)utf8str.size(), NULL, 0);
	wstring ws_translated_str(size_required, 0);
	MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(),
		&ws_translated_str[0], size_required);
	return ws_translated_str;
}

inline string wchar_to_utf8(const wstring& wstr)
{
	if (wstr.empty())
		return string();
	int size_required = WideCharToMultiByte(
		CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
	string utf8_translated_str(size_required, 0);
	WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(),
		&utf8_translated_str[0], size_required, NULL, NULL);
	return utf8_translated_str;
}

SSIZE_T pread(HANDLE hfile, char* src, size_t num_bytes, uint64_t offset)
{
	assert(num_bytes <= (numeric_limits<DWORD>::max)());
	OVERLAPPED overlapped = {0};
	ULARGE_INTEGER offset_union;
	offset_union.QuadPart = offset;

	overlapped.Offset = offset_union.LowPart;
	overlapped.OffsetHigh = offset_union.HighPart;
	overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);

	if (NULL == overlapped.hEvent)
		return -1;

	SSIZE_T result = 0;

	unsigned long bytes_read = 0;
	DWORD last_error = ERROR_SUCCESS;

	BOOL read_result = ::ReadFile(hfile, src, static_cast<DWORD>(num_bytes),
	                            &bytes_read, &overlapped);
	if (TRUE == read_result)
	{
		result = bytes_read;
	}
	else if ((FALSE == read_result) &&
	         ((last_error = GetLastError()) != ERROR_IO_PENDING))
	{
		result = (last_error == ERROR_HANDLE_EOF) ? 0 : -1;
	}
	else
	{
		if (ERROR_IO_PENDING == last_error)
		{
			BOOL overlapped_result =
				::GetOverlappedResult(hfile, &overlapped, &bytes_read, TRUE);
			if (FALSE == overlapped_result)
				result = (::GetLastError() == ERROR_HANDLE_EOF) ? 0 : -1;
			else
				result = bytes_read;
		}
	}

	::CloseHandle(overlapped.hEvent);

	return result;
}

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

	error_condition read(uint64_t offset, size_t n, string_view* result, char* scratch) const override
	{
		char* dst = scratch;
		error_condition ec {};
		while (n > 0 && !ec)
		{
			auto r = pread(m_hfile, dst, n, offset);
			if (r > 0)
			{
				offset += r;
				dst += r;
				n -= r;
			}
			else if (r == 0)
			{
				ec = make_error_condition(ShogunErrc::OutOfRange);
			}
			else if (errno == EINTR || errno == EAGAIN) {
				// retry
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

	error_condition append(string_view data) override
	{
		DWORD bytes_written = 0;
		DWORD data_size = static_cast<DWORD>(data.size());
		BOOL write_result =
		    ::WriteFile(m_hfile, data.data(), data_size, &bytes_written, NULL);
		if (FALSE == write_result)
			return generic_category().default_error_condition(::GetLastError());

		assert(size_t(bytes_written) == data.size());
		return {};
	}

	error_condition close() override
	{
		assert(INVALID_HANDLE_VALUE != m_hfile);

		auto r = flush();
		if (r)
			return r;

		if (FALSE == ::CloseHandle(m_hfile))
			return generic_category().default_error_condition(::GetLastError());

		m_hfile = INVALID_HANDLE_VALUE;
		return {};
	}

	error_condition flush() override
	{
		if (FALSE == ::FlushFileBuffers(m_hfile))
			return generic_category().default_error_condition(::GetLastError());
		return {};
	}

	error_condition sync() override
	{
		return flush();
	}

private:
	string m_filename;
	HANDLE m_hfile;
};

error_condition WindowsFileSystem::new_random_access_file(
	const string& fname, unique_ptr<RandomAccessFile>* file) const
{
	auto translated_fname = translate_name(fname);
	auto ws_translated_fname = utf8_to_wchar(translated_fname);
	file->reset(nullptr);

	DWORD file_flags = FILE_ATTRIBUTE_READONLY | FILE_FLAG_OVERLAPPED;
	DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

	HANDLE hfile =
	  ::CreateFileW(ws_translated_fname.c_str(), GENERIC_READ, share_mode, NULL,
	                OPEN_EXISTING, file_flags, NULL);

	if (INVALID_HANDLE_VALUE == hfile)
		return generic_category().default_error_condition(::GetLastError());

	file->reset(new WindowsRandomAccessFile(translated_fname, hfile));
	return {};
}

error_condition WindowsFileSystem::new_writable_file(
	const string& fname, unique_ptr<WritableFile>* file) const
{
	auto translated_fname = translate_name(fname);
	auto ws_translated_fname = utf8_to_wchar(translated_fname);
	file->reset(nullptr);

	DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
	HANDLE hfile =
	  ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
	                NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	if (INVALID_HANDLE_VALUE == hfile)
		return generic_category().default_error_condition(::GetLastError());

	file->reset(new WindowsWritableFile(translated_fname, hfile));
	return {};
}

error_condition WindowsFileSystem::new_appendable_file(
	const string& fname, unique_ptr<WritableFile>* file) const
{
	auto translated_fname = translate_name(fname);
	auto ws_translated_fname = utf8_to_wchar(translated_fname);
	file->reset(nullptr);

	DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
	HANDLE hfile =
	  ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
	                NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	if (INVALID_HANDLE_VALUE == hfile)
		return generic_category().default_error_condition(::GetLastError());

	UniqueCloseHandlePtr file_guard(hfile, CloseHandleFunc);

	DWORD file_ptr = ::SetFilePointer(hfile, NULL, NULL, FILE_END);
	if (INVALID_SET_FILE_POINTER == file_ptr)
		return generic_category().default_error_condition(::GetLastError());

	file->reset(new WindowsWritableFile(translated_fname, hfile));
	file_guard.release();

	return {};
}

error_condition WindowsFileSystem::file_exists(const string& fname) const
{
	constexpr int kOk = 0;
	auto ws_translated_fname = utf8_to_wchar(translate_name(fname));
	if (_waccess(ws_translated_fname.c_str(), kOk) == 0)
		return {};

	return generic_category().default_error_condition(errno);
}

error_condition WindowsFileSystem::delete_file(const string& fname) const
{
	auto file_name = utf8_to_wchar(fname);
	if (_wunlink(file_name.c_str()) != 0)
		return generic_category().default_error_condition(errno);
	return {};
}

error_condition WindowsFileSystem::create_dir(const string& name) const
{
	auto ws_name = utf8_to_wchar(name);
	if (_wmkdir(ws_name.c_str()) != 0)
		return generic_category().default_error_condition(errno);

	return {};
}

error_condition WindowsFileSystem::delete_dir(const string& dirname) const
{
	auto ws_name = utf8_to_wchar(dirname);
	if (_wrmdir(ws_name.c_str()) != 0)
		return generic_category().default_error_condition(errno);

	return {};
}

error_condition WindowsFileSystem::rename_file(const string& src, const string& target) const
{
	auto ws_translated_src = utf8_to_wchar(translate_name(src));
	auto ws_translated_target = utf8_to_wchar(translate_name(target));
	if (!::MoveFileExW(ws_translated_src.c_str(), ws_translated_target.c_str(),
	             MOVEFILE_REPLACE_EXISTING))
		return generic_category().default_error_condition(::GetLastError());

	return {};
}

int64_t WindowsFileSystem::get_file_size(const string& name) const
{
	auto translated_fname = translate_name(name);
	auto ws_translated_dir = utf8_to_wchar(translated_fname);
	WIN32_FILE_ATTRIBUTE_DATA attrs;
	if (TRUE == ::GetFileAttributesExW(ws_translated_dir.c_str(),
		GetFileExInfoStandard, &attrs))
	{
		ULARGE_INTEGER file_size;
		file_size.HighPart = attrs.nFileSizeHigh;
		file_size.LowPart = attrs.nFileSizeLow;
		return file_size.QuadPart;
	}
	else
	{
		return -1; //generic_category().default_error_condition(::GetLastError());
	}
}

error_condition WindowsFileSystem::is_directory(const string& name) const
{
	auto ws_translated_dir = utf8_to_wchar(translate_name(name));
	if (PathIsDirectoryW(ws_translated_dir.c_str()))
		return {};
	return make_error_condition(errc::io_error);
}

error_condition WindowsFileSystem::get_children(const string& dir,
	vector<string>* result) const
{
	auto translated_fname = translate_name(dir);
	auto ws_translated_dir = utf8_to_wchar(translated_fname);
	result->clear();

	auto pattern = ws_translated_dir;
	if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/')
		pattern += L"\\*";
	else
		pattern += L'*';

	WIN32_FIND_DATAW find_data;
	HANDLE find_handle = ::FindFirstFileW(pattern.c_str(), &find_data);
	if (find_handle == INVALID_HANDLE_VALUE)
		return generic_category().default_error_condition(::GetLastError());

	do
	{
		auto file_name = wchar_to_utf8(find_data.cFileName);
		const string_view basename = file_name;
		if (basename != "." && basename != "..")
			result->push_back(file_name);
	} while (::FindNextFileW(find_handle, &find_data));

	if (!::FindClose(find_handle))
		return generic_category().default_error_condition(::GetLastError());

	return {};
}


error_condition WindowsFileSystem::get_paths(const string& pattern,
				vector<string>* results) const
{
	//FIXME
	return {};
}

REGISTER_FILE_SYSTEM("", WindowsFileSystem);
REGISTER_FILE_SYSTEM("file", LocalWindowsFileSystem);

#endif
