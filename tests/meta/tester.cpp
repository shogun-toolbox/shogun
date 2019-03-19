#include <shogun/base/init.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/lib/DynamicObjectArray.h>

#include <iostream>
#include <sstream>

using namespace shogun;
using namespace std;

int main(int argc, const char *argv[])
{
    if (argc != 6)
        cout << "Usage: tester REL_DIR NAME TARGET GENERATED_RESULTS_DIR REFERENCE_RESULTS_DIR" << endl;

    string rel_dir = argv[1];
    string name = argv[2];
    string target = argv[3];
    string generated_results_dir = argv[4];
    string reference_results_dir = argv[5];

    stringstream os;
    os << generated_results_dir << "/" << target << "/" << rel_dir << "/" << name;
    string fname_full_generated = os.str();
    os.str("");
    os << reference_results_dir << "/" << rel_dir << "/" << name;
    string fname_full_reference = os.str();

    init_shogun_with_defaults();

    CSerializableAsciiFile* f = new CSerializableAsciiFile(fname_full_generated.c_str(), 'r');
    CSerializableAsciiFile* f_ref = new CSerializableAsciiFile(fname_full_reference.c_str(), 'r');
    SG_REF(f);
    SG_REF(f_ref);

    auto a = some<CDynamicObjectArray>();
    auto a_ref = some<CDynamicObjectArray>();

    if (!a->load_serializable(f))
		SG_SERROR(
		    "Error while loading the test input from %s\n",
		    fname_full_generated.c_str());
	if (!a_ref->load_serializable(f_ref))
		SG_SERROR(
		    "Error while loading the reference input from %s\n",
		    fname_full_reference.c_str());

	// allow for lossy serialization format
	set_global_fequals_epsilon(1e-6);
	bool loaded_equals_ref = a->equals(a_ref);

	// it is unlikely that the above is true while this is false, however, we
	// check
	bool ref_equals_loaded = a_ref->equals(a);

	// print comparison output only if different as it it slow
	if (!loaded_equals_ref || !ref_equals_loaded)
	{
		a->get_global_io()->set_loglevel(MSG_DEBUG);
		if (!loaded_equals_ref)
		{
			SG_SDEBUG(
			    "Test input different from reference input for %s\n",
			    a->get_name());
			a->equals(a_ref);
		}
		else
		{
			SG_SDEBUG(
			    "Reference input different from test input for %s.\n",
			    a->get_name());
			a_ref->equals(a);
		}
		SG_SDEBUG(
		    "For details, run: diff %s %s\n", fname_full_generated.c_str(),
		    fname_full_reference.c_str());
	}

	SG_UNREF(f);
	SG_UNREF(f_ref);

	exit_shogun();
	return !(loaded_equals_ref && ref_equals_loaded);
}

