/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/Serializer.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/FileOutputStream.h>

using namespace shogun;
using namespace shogun::io;

CSerializer::CSerializer() : CSGObject(), m_stream(empty<COutputStream>())
{
}

CSerializer::~CSerializer()
{
}

void CSerializer::attach(Some<COutputStream> stream)
{
    m_stream = stream;
}

Some<COutputStream> CSerializer::stream() const
{
	REQUIRE(m_stream, "Serializer has no stream, attach() it to a stream");
	return m_stream;
}

void shogun::io::pre_serialize(CSGObject* obj) noexcept(false)
{
	obj->save_serializable_pre();

	if (!obj->get_save_serializable_pre())
	{
		SG_SERROR("%s::save_serializable_pre(): Implementation "
			"error: BASE_CLASS::SAVE_SERIALIZABLE_PRE() not "
			"called!", obj->get_name());
	}
}

void shogun::io::post_serialize(CSGObject* obj) noexcept(false)
{
	obj->save_serializable_post();

	if (!obj->get_save_serializable_post())
	{
		SG_SERROR("%s::save_serializable_post(): Implementation "
			"error: BASE_CLASS::SAVE_SERIALIZABLE_POST() not "
			"called!", obj->get_name());
	}
}

void shogun::io::serialize(const std::string& _path, CSGObject* _obj, CSerializer* _serializer)
{
	auto fs = io::FileSystemRegistry::instance();
	std::error_condition ec;
	std::unique_ptr<io::WritableFile> file;
	if ((ec = fs->new_writable_file(_path, &file)))
		throw to_system_error(ec);

	auto fos = some<io::CFileOutputStream>(file.get());
	_serializer->attach(fos);
	_serializer->write(wrap(_obj));
}