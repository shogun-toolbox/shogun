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

Deserializer::Deserializer() : m_stream(empty<CInputStream>())
{
}

Deserializer::~Deserializer()
{
}

void Deserializer::attach(Some<CInputStream> stream)
{
	m_stream = stream;
}

Some<CInputStream> Deserializer::stream() const
{
	REQUIRE(m_stream, "Deserializer has no stream, attach() it to a stream");
	return m_stream;
}

void shogun::io::pre_deserialize(CSGObject* obj) noexcept(false)
{
	obj->load_serializable_pre();

	if (!obj->get_load_serializable_pre())
	{
		SG_SERROR("%s::load_serializable_pre(): Implementation "
			"error: BASE_CLASS::LOAD_SERIALIZABLE_PRE() not "
			"called!", obj->get_name());
	}
}

void shogun::io::post_deserialize(CSGObject* obj) noexcept(false)
{
	obj->load_serializable_post();

	if (!obj->get_load_serializable_post())
	{
		SG_SERROR("%s::load_serializable_post(): Implementation "
			"error: BASE_CLASS::LOAD_SERIALIZABLE_POST() not "
			"called!", obj->get_name());
	}
}

CSGObject* shogun::io::deserialize(const std::string& _path, Deserializer* _deser)
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
