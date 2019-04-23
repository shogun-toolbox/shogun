#include <shogun/base/ShogunEnv.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/io/fs/Path.h>

#include <iostream>

using namespace shogun;
using namespace std;

static std::shared_ptr<SGObject> load_object(const string& fname)
{
	auto fs = env();
	auto ec = fs->file_exists(fname);
	if (ec)
		throw io::to_system_error(ec);
	std::unique_ptr<io::RandomAccessFile> f;
	ec = fs->new_random_access_file(fname, &f);
	if (ec)
		throw io::to_system_error(ec);
	auto fis = std::make_shared<io::FileInputStream>(f.get());
	auto deserializer = std::make_unique<io::JsonDeserializer>();
	deserializer->attach(fis);
	return deserializer->read_object();
}

int main(int argc, const char *argv[])
{
	if (argc != 6)
		cout << "Usage: tester REL_DIR NAME TARGET GENERATED_RESULTS_DIR REFERENCE_RESULTS_DIR" << endl;

	string rel_dir = argv[1];
	string name = argv[2];
	string target = argv[3];
	string generated_results_dir = argv[4];
	string reference_results_dir = argv[5];

	auto fname_full_generated = io::join_path(generated_results_dir, target, rel_dir, name);
	auto fname_full_reference = io::join_path(reference_results_dir, rel_dir, name);
	auto a = load_object(fname_full_generated);
	auto a_ref = load_object(fname_full_reference);

	// allow for lossy serialization format
	env()->set_global_fequals_epsilon(1e-6);
	bool loaded_equals_ref = a->equals(a_ref);

	// it is unlikely that the above is true while this is false, however, we
	// check
	bool ref_equals_loaded = a_ref->equals(a);

	// print comparison output only if different as it it slow
	if (!loaded_equals_ref || !ref_equals_loaded)
	{
		env()->io()->set_loglevel(io::MSG_DEBUG);
		if (!loaded_equals_ref)
		{
			SG_DEBUG(
				"Test input different from reference input for {}\n",
				a->get_name());
			a->equals(a_ref);
		}
		else
		{
			SG_DEBUG(
				"Reference input different from test input for {}.\n",
				a->get_name());
			a_ref->equals(a);
		}
		SG_DEBUG(
			"For details, run: diff {} {}\n", fname_full_generated.c_str(),
			fname_full_reference.c_str());
	}

	return !(loaded_equals_ref && ref_equals_loaded);
}

