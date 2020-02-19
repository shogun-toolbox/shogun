#ifndef __WINDOWS_WCHAR_H__
#define __WINDOWS_WCHAR_H__

#include <shogun/lib/common.h>
#define NOMINMAX
#include <Windows.h>
#include <string>
#include <string_view>

namespace shogun
{
    inline std::wstring utf8_to_wchar(std::string_view utf8str)
    {
        int size_required = MultiByteToWideChar(
            CP_UTF8, 0, utf8str.data(),
            (int)utf8str.size(), NULL, 0);
        std::wstring ws_translated_str(size_required, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8str.data(), (int)utf8str.size(),
            &ws_translated_str[0], size_required);
        return ws_translated_str;
    }

    inline std::wstring utf8_to_wchar(const std::string& utf8str)
    {
        int size_required = MultiByteToWideChar(
            CP_UTF8, 0, utf8str.c_str(),
            (int)utf8str.size(), NULL, 0);
        std::wstring ws_translated_str(size_required, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(),
            &ws_translated_str[0], size_required);
        return ws_translated_str;
    }

    inline std::string wchar_to_utf8(const std::wstring& wstr)
    {
        if (wstr.empty())
            return std::string();
        int size_required = WideCharToMultiByte(
            CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
        std::string utf8_translated_str(size_required, 0);
        WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(),
            &utf8_translated_str[0], size_required, NULL, NULL);
        return utf8_translated_str;
    }
} // namespace shogun
#endif
