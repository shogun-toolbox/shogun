/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma, Viktor Gal
 */

#ifndef _LIBRARY_H_
#define _LIBRARY_H_

#include <shogun/base/manifest.h>
#include <string_view>

namespace shogun
{

	namespace internal
	{
		/** @brief
		 * Handles loading, calling and closing of plugins from shared object files.
		 */
		class LibraryHandle;
	}

	/** @brief
	 * Provides an API for loading plugins as objects of this class
	 * and accessing Manifest of the loaded plugins.
	 * Uses LibraryHandle under the hood.
	 */
	class SHOGUN_EXPORT Library
	{
	public:
		/** Constructor to initialize library
		 * @param filename name of shared object file
		 */
		Library(std::string_view filename);

		/** Copy constructor
		 * @param other library object to be copied
		 */
		Library(const Library& other);

		/** Class Assignment operator
		 * @param other library object to be assigned
		 */
		Library& operator=(const Library& other);

		/** Equality operator
		 * @param first first Library
		 * @param second second Library
		 */
		SHOGUN_EXPORT friend bool operator==(const Library& first, const Library& second);

		/** Inequality operator
		 * @param first first Library
		 * @param second second Library
		 */
		SHOGUN_EXPORT friend bool operator!=(const Library& first, const Library& second);

		/** Destructor */
		~Library();

		/** @return manifest of loaded library */
		Manifest manifest();

		/** @return name of function that accesses Manifest
		 * of loaded library.
		 */
		static std::string_view get_manifest_accessor_name()
		{
			return kManifestAccessorName;
		}

		SHOGUN_EXPORT friend void unload_library(Library&& lib);

	protected:
		void close();

	private:
		std::shared_ptr<internal::LibraryHandle> m_handle;
		static constexpr std::string_view kManifestAccessorName = "shogunManifest";
	};

	/** Loads a plugin into a library object.
	 * @param filename name of shared object file
	 * @return library object of loaded plugin
	 */
	SHOGUN_EXPORT Library load_library(std::string_view filename);

	/** Unload a plugin from the process mem space.
	 * @param lib library
	 */
	SHOGUN_EXPORT void unload_library(Library&& lib);
}

#endif //_LIBRARY_H_
