/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben (modified from
 * https://docs.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=vs-2019)
 */

#include "CPUArch.h"
#include <shogun/io/SGIO.h>

using namespace shogun;
using namespace shogun::graph;

#ifdef _MSC_VER
#include <array>
#include <bitset>
#include <intrin.h>
#include <string>
#include <vector>

class InstructionSet
{
	// forward declarations
	class InstructionSet_Internal;

public:
	// getters
	static std::string Vendor(void)
	{
		return CPU_Rep.vendor_;
	}
	static std::string Brand(void)
	{
		return CPU_Rep.brand_;
	}

	static bool SSE3(void)
	{
		return CPU_Rep.f_1_ECX_[0];
	}
	static bool PCLMULQDQ(void)
	{
		return CPU_Rep.f_1_ECX_[1];
	}
	static bool MONITOR(void)
	{
		return CPU_Rep.f_1_ECX_[3];
	}
	static bool SSSE3(void)
	{
		return CPU_Rep.f_1_ECX_[9];
	}
	static bool FMA(void)
	{
		return CPU_Rep.f_1_ECX_[12];
	}
	static bool CMPXCHG16B(void)
	{
		return CPU_Rep.f_1_ECX_[13];
	}
	static bool SSE41(void)
	{
		return CPU_Rep.f_1_ECX_[19];
	}
	static bool SSE42(void)
	{
		return CPU_Rep.f_1_ECX_[20];
	}
	static bool MOVBE(void)
	{
		return CPU_Rep.f_1_ECX_[22];
	}
	static bool POPCNT(void)
	{
		return CPU_Rep.f_1_ECX_[23];
	}
	static bool AES(void)
	{
		return CPU_Rep.f_1_ECX_[25];
	}
	static bool XSAVE(void)
	{
		return CPU_Rep.f_1_ECX_[26];
	}
	static bool OSXSAVE(void)
	{
		return CPU_Rep.f_1_ECX_[27];
	}
	static bool AVX(void)
	{
		return CPU_Rep.f_1_ECX_[28];
	}
	static bool F16C(void)
	{
		return CPU_Rep.f_1_ECX_[29];
	}
	static bool RDRAND(void)
	{
		return CPU_Rep.f_1_ECX_[30];
	}

	static bool MSR(void)
	{
		return CPU_Rep.f_1_EDX_[5];
	}
	static bool CX8(void)
	{
		return CPU_Rep.f_1_EDX_[8];
	}
	static bool SEP(void)
	{
		return CPU_Rep.f_1_EDX_[11];
	}
	static bool CMOV(void)
	{
		return CPU_Rep.f_1_EDX_[15];
	}
	static bool CLFSH(void)
	{
		return CPU_Rep.f_1_EDX_[19];
	}
	static bool MMX(void)
	{
		return CPU_Rep.f_1_EDX_[23];
	}
	static bool FXSR(void)
	{
		return CPU_Rep.f_1_EDX_[24];
	}
	static bool SSE(void)
	{
		return CPU_Rep.f_1_EDX_[25];
	}
	static bool SSE2(void)
	{
		return CPU_Rep.f_1_EDX_[26];
	}

	static bool FSGSBASE(void)
	{
		return CPU_Rep.f_7_EBX_[0];
	}
	static bool BMI1(void)
	{
		return CPU_Rep.f_7_EBX_[3];
	}
	static bool HLE(void)
	{
		return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4];
	}
	static bool AVX2(void)
	{
		return CPU_Rep.f_7_EBX_[5];
	}
	static bool BMI2(void)
	{
		return CPU_Rep.f_7_EBX_[8];
	}
	static bool ERMS(void)
	{
		return CPU_Rep.f_7_EBX_[9];
	}
	static bool INVPCID(void)
	{
		return CPU_Rep.f_7_EBX_[10];
	}
	static bool RTM(void)
	{
		return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11];
	}
	static bool AVX512F(void)
	{
		return CPU_Rep.f_7_EBX_[16];
	}
	static bool RDSEED(void)
	{
		return CPU_Rep.f_7_EBX_[18];
	}
	static bool ADX(void)
	{
		return CPU_Rep.f_7_EBX_[19];
	}
	static bool AVX512PF(void)
	{
		return CPU_Rep.f_7_EBX_[26];
	}
	static bool AVX512ER(void)
	{
		return CPU_Rep.f_7_EBX_[27];
	}
	static bool AVX512CD(void)
	{
		return CPU_Rep.f_7_EBX_[28];
	}
	static bool SHA(void)
	{
		return CPU_Rep.f_7_EBX_[29];
	}

	static bool PREFETCHWT1(void)
	{
		return CPU_Rep.f_7_ECX_[0];
	}

	static bool LAHF(void)
	{
		return CPU_Rep.f_81_ECX_[0];
	}
	static bool LZCNT(void)
	{
		return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5];
	}
	static bool ABM(void)
	{
		return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5];
	}
	static bool SSE4a(void)
	{
		return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6];
	}
	static bool XOP(void)
	{
		return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11];
	}
	static bool TBM(void)
	{
		return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21];
	}

	static bool SYSCALL(void)
	{
		return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11];
	}
	static bool MMXEXT(void)
	{
		return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22];
	}
	static bool RDTSCP(void)
	{
		return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27];
	}
	static bool _3DNOWEXT(void)
	{
		return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30];
	}
	static bool _3DNOW(void)
	{
		return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31];
	}

private:
	static const InstructionSet_Internal CPU_Rep;

	class InstructionSet_Internal
	{
	public:
		InstructionSet_Internal()
		    : nIds_{0}, nExIds_{0}, isIntel_{false}, isAMD_{false}, f_1_ECX_{0},
		      f_1_EDX_{0}, f_7_EBX_{0}, f_7_ECX_{0}, f_81_ECX_{0}, f_81_EDX_{0},
		      data_{}, extdata_{}
		{
			// int cpuInfo[4] = {-1};
			std::array<int, 4> cpui;

			// Calling __cpuid with 0x0 as the function_id argument
			// gets the number of the highest valid function ID.
			__cpuid(cpui.data(), 0);
			nIds_ = cpui[0];

			for (int i = 0; i <= nIds_; ++i)
			{
				__cpuidex(cpui.data(), i, 0);
				data_.push_back(cpui);
			}

			// Capture vendor string
			char vendor[0x20];
			memset(vendor, 0, sizeof(vendor));
			*reinterpret_cast<int*>(vendor) = data_[0][1];
			*reinterpret_cast<int*>(vendor + 4) = data_[0][3];
			*reinterpret_cast<int*>(vendor + 8) = data_[0][2];
			vendor_ = vendor;
			if (vendor_ == "GenuineIntel")
			{
				isIntel_ = true;
			}
			else if (vendor_ == "AuthenticAMD")
			{
				isAMD_ = true;
			}

			// load bitset with flags for function 0x00000001
			if (nIds_ >= 1)
			{
				f_1_ECX_ = data_[1][2];
				f_1_EDX_ = data_[1][3];
			}

			// load bitset with flags for function 0x00000007
			if (nIds_ >= 7)
			{
				f_7_EBX_ = data_[7][1];
				f_7_ECX_ = data_[7][2];
			}

			// Calling __cpuid with 0x80000000 as the function_id argument
			// gets the number of the highest valid extended ID.
			__cpuid(cpui.data(), 0x80000000);
			nExIds_ = cpui[0];

			char brand[0x40];
			memset(brand, 0, sizeof(brand));

			for (int i = 0x80000000; i <= nExIds_; ++i)
			{
				__cpuidex(cpui.data(), i, 0);
				extdata_.push_back(cpui);
			}

			// load bitset with flags for function 0x80000001
			if (nExIds_ >= 0x80000001)
			{
				f_81_ECX_ = extdata_[1][2];
				f_81_EDX_ = extdata_[1][3];
			}

			// Interpret CPU brand string if reported
			if (nExIds_ >= 0x80000004)
			{
				memcpy(brand, extdata_[2].data(), sizeof(cpui));
				memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
				memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
				brand_ = brand;
			}
		};

		int nIds_;
		int nExIds_;
		std::string vendor_;
		std::string brand_;
		bool isIntel_;
		bool isAMD_;
		std::bitset<32> f_1_ECX_;
		std::bitset<32> f_1_EDX_;
		std::bitset<32> f_7_EBX_;
		std::bitset<32> f_7_ECX_;
		std::bitset<32> f_81_ECX_;
		std::bitset<32> f_81_EDX_;
		std::vector<std::array<int, 4>> data_;
		std::vector<std::array<int, 4>> extdata_;
	};
};

// Initialize static member data
const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;
#endif

CPUArch::CPUArch() : instructions(SIMD::NONE)
{
#if defined(__GNUC__) || defined(__CLANG__)
	__builtin_cpu_init();
	if (__builtin_cpu_supports("sse"))
		enable_simd(instructions, SIMD::SSE);
	if (__builtin_cpu_supports("sse2"))
		enable_simd(instructions, SIMD::SSE2);
	if (__builtin_cpu_supports("sse3"))
		enable_simd(instructions, SIMD::SSE3);
	if (__builtin_cpu_supports("sse4.1"))
		enable_simd(instructions, SIMD::SSE4_1);
	if (__builtin_cpu_supports("sse4.2"))
		enable_simd(instructions, SIMD::SSE4_2);
	if (__builtin_cpu_supports("avx"))
		enable_simd(instructions, SIMD::AVX);
	if (__builtin_cpu_supports("avx2"))
		enable_simd(instructions, SIMD::AVX2);
	if (__builtin_cpu_supports("avx512f"))
		enable_simd(instructions, SIMD::AVX512F);
#elif defined(_MSC_VER)
	if (InstructionSet::SSE)
		enable_simd(instructions, SIMD::SSE);
	if (InstructionSet::SSE2)
		enable_simd(instructions, SIMD::SSE2);
	if (InstructionSet::SSE3)
		enable_simd(instructions, SIMD::SSE3);
	if (InstructionSet::SSE41)
		enable_simd(instructions, SIMD::SSE4_1);
	if (InstructionSet::SSE42)
		enable_simd(instructions, SIMD::SSE4_2);
	if (InstructionSet::AVX)
		enable_simd(instructions, SIMD::AVX);
	if (InstructionSet::AVX2)
		enable_simd(instructions, SIMD::AVX2);
	if (InstructionSet::AVX512F)
		enable_simd(instructions, SIMD::AVX512F);
#else
	io::warn("There is no support for runtime CPU architecture detection with "
	         "the provided compiler. Defaulting to SSE2 instructions.")
	    enable_simd(instructions, SIMD::SSE2);
#endif
	io::info("Found SSE:     {}", has_sse());
	io::info("Found SSE2:    {}", has_sse2());
	io::info("Found SSE3:    {}", has_sse3());
	io::info("Found SSE4.1:  {}", has_sse4_1());
	io::info("Found SSE4.2:  {}", has_sse4_2());
	io::info("Found AVX:     {}", has_avx());
	io::info("Found AVX2:    {}", has_avx2());
	io::info("Found AVX512F: {}", has_avx512f());
}

CPUArch* CPUArch::instance()
{
	static CPUArch architecture{};
	return &architecture;
}