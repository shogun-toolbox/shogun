/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Pan Deng, Heiko Strathmann, Soeren Sonnenburg, Giovanni De Toni,
 *          Yuyu Zhang, Viktor Gal, Sergey Lisitsyn
 */

#ifndef __SHOGUNENV_H__
#define __SHOGUNENV_H__

#include <shogun/lib/common.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/Version.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/fs/FileSystemRegistry.h>

#include <memory>

namespace shogun
{
	class SGIO;
	class SGLinalg;
	class Signal;

	class ShogunEnv : public io::FileSystemRegistry, public Parallel, public Version
	{
	public:
		SG_DELETE_COPY_AND_ASSIGN(ShogunEnv);

		/** Returns a singleton instance of the class
		 */
		static ShogunEnv* instance();

		/** Destructor
		 */
		~ShogunEnv();

		/** set the global io object
		 *
		 * @param io io object to use
		 */
		void set_global_io(const std::shared_ptr<SGIO>& io);

		/** get the global io object
		 *
		 * @return io object
		 */
		SGIO* io();

		/** @return the globally over-ridden floating point epsilon for
		 * CMath::fequals
		 */
		float64_t fequals_epsilon();

		/** Globally over-ride the floating point epsilon for CMath::fequals.
		 * Hack required for CSGObject::equals checks for certain serialization
		 * formats.
		 * @param fequals_epsilon new epsilon to use
		 */
		void set_global_fequals_epsilon(float64_t fequals_epsilon);

		/** @return whether global linient check for CMath::fequals is enabled
		 */
		bool fequals_tolerant();

		/** Globally enable linient check for CMath::fequals.
		 * Hack required for CSGObject::equals checks for certain serialization
		 * formats.
		 * @param fequals_tolerant whether or not to use tolerant check
		 */
		void set_global_fequals_tolerant(bool fequals_tolerant);

#ifndef SWIG // SWIG should skip this part
		/** get the global linalg library object
		 *
		 * @return linalg object
		 */
		SGLinalg* linalg();
#endif

		/** get the global singnal handler object
		 *
		 * @return linalg object
		 */
		Signal* signal();

	private:
		/** Default constructor
		 */
		ShogunEnv();

		/** Checks environment variables and modifies global objects
		 */
		void init_from_env();

		std::shared_ptr<SGIO> sg_io;
		std::unique_ptr<Signal> sg_signal;
		std::unique_ptr<SGLinalg> sg_linalg;
		float64_t sg_fequals_epsilon;
		bool sg_fequals_tolerant;
	};

	static inline ShogunEnv* env()
	{
		return ShogunEnv::instance();
	}
} // namespace shogun

#endif // __SHOGUNENV_H__
