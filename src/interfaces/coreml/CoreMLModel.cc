#include "CoreMLModel.h"

#include <fstream>

#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "format/Model.pb.h"

using namespace shogun::coreml;

CoreMLModel::CoreMLModel():
	m_spec(std::make_shared<CoreML::Specification::Model>())
{
}

CoreMLModel::CoreMLModel(std::shared_ptr<CoreML::Specification::Model> spec):
	m_spec(spec)
{
}

CoreMLModel::~CoreMLModel()
{
	m_spec.reset();
}

void CoreMLModel::save(const std::string& filename) const
{
	std::fstream out(filename, std::ios::binary | std::ios::out);
	this->save(out);
	out.close();
}

void CoreMLModel::save(std::ostream& out) const
{
	::google::protobuf::io::OstreamOutputStream pb_out(&out);
	if (!m_spec->SerializeToZeroCopyStream(&pb_out))
		throw std::runtime_error("could not save");
}

void CoreMLModel::set_short_description(const std::string& desc)
{
	m_spec->mutable_description()->mutable_metadata()->set_shortdescription(desc);
}

void CoreMLModel::set_license(const std::string& license)
{
	m_spec->mutable_description()->mutable_metadata()->set_license(license);
}

void CoreMLModel::set_author(const std::string& author)
{
	m_spec->mutable_description()->mutable_metadata()->set_author(author);
}
