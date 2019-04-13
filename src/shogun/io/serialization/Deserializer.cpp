/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/Deserializer.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/FileInputStream.h>

using namespace shogun;
using namespace shogun::io;

CDeserializer::CDeserializer() : CSGObject(), m_stream(empty<CInputStream>())
{
}

CDeserializer::~CDeserializer()
{
}

void CDeserializer::attach(Some<CInputStream> stream)
{
	m_stream = stream;
}

Some<CInputStream> CDeserializer::stream() const
{
	REQUIRE(m_stream, "Deserializer has no stream, attach() it to a stream");
	return m_stream;
}

CSGObject* shogun::io::deserialize(const std::string& _path, CDeserializer* _deser)
{
	auto fs = io::FileSystemRegistry::instance();
	std::error_condition ec;
	if ((ec = fs->file_exists(_path)))
		throw to_system_error(ec);

	std::unique_ptr<io::RandomAccessFile> raf;
	if ((ec = fs->new_random_access_file(_path, &raf)))
		throw to_system_error(ec);

	auto fis = some<io::CFileInputStream>(raf.get());
	_deser->attach(fis);
	return _deser->read_object().get();
}
