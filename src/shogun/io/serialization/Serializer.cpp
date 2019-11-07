/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/base/ShogunEnv.h>
#include <shogun/io/serialization/Serializer.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/FileOutputStream.h>

#include <utility>

using namespace shogun;
using namespace shogun::io;

Serializer::Serializer() : SGObject()
{
}

Serializer::~Serializer()
{
}

void Serializer::attach(std::shared_ptr<OutputStream> stream)
{
    m_stream = std::move(stream);
}

std::shared_ptr<OutputStream> Serializer::stream() const
{
	require(m_stream, "Serializer has no stream, attach() it to a stream");
	return m_stream;
}

void shogun::io::pre_serialize(const std::shared_ptr<SGObject>& obj) noexcept(false)
{
	obj->save_serializable_pre();

	if (!obj->get_save_serializable_pre())
	{
		error("{}::save_serializable_pre(): Implementation "
			"error: BASE_CLASS::SAVE_SERIALIZABLE_PRE() not "
			"called!", obj->get_name());
	}
}

void shogun::io::post_serialize(const std::shared_ptr<SGObject>& obj) noexcept(false)
{
	obj->save_serializable_post();

	if (!obj->get_save_serializable_post())
	{
		error("{}::save_serializable_post(): Implementation "
			"error: BASE_CLASS::SAVE_SERIALIZABLE_POST() not "
			"called!", obj->get_name());
	}
}

void shogun::io::serialize(const std::string& _path, std::shared_ptr<SGObject> _obj, const std::shared_ptr<Serializer>& _serializer)
{
	auto fs = env();
	std::error_condition ec;
	std::unique_ptr<io::WritableFile> file;
	if ((ec = fs->new_writable_file(_path, &file)))
		throw to_system_error(ec);

	auto fos = std::make_shared<io::FileOutputStream>(file.get());
	_serializer->attach(fos);
	_serializer->write(std::move(_obj));
}
