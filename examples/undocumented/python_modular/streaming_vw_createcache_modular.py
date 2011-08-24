from modshogun import StreamingVwFile
from modshogun import StreamingVwCacheFile
from modshogun import T_SVMLIGHT
from modshogun import StreamingVwFeatures
from modshogun import VowpalWabbit

def create_cache():
    """Creates a binary cache from an ascii data file."""

    # Open the input file as a StreamingVwFile
    input_file = StreamingVwFile("../data/fm_train_sparsereal.dat")
    # Default file name will be vw_cache.dat.cache
    input_file.set_write_to_cache(True)

    # Tell VW that the file is in SVMLight format
    # Supported types are T_DENSE, T_SVMLIGHT and T_VW
    input_file.set_parser_type(T_SVMLIGHT)

    # Create a StreamingVwFeatures object, `True' indicating the examples are labelled
    features = StreamingVwFeatures(input_file, True, 1024)

    # Create a VW object from the features
    vw = VowpalWabbit(features)
    vw.set_no_training(True)
    
    # Train (in this case does nothing but run over all examples)
    vw.train()

def train_from_cache():
    """Train using the generated cache file."""
    
    # Open the input cache file as a StreamingVwCacheFile
    input_file = StreamingVwCacheFile("vw_cache.dat.cache");

    # The rest is exactly as for normal input
    features = StreamingVwFeatures(input_file, True, 1024);
    vw = VowpalWabbit(features)
    vw.train()

if __name__ == "__main__":
    print "Creating cache..."
    create_cache()
    print "Done."
    print

    print "Training using the cache file..."
    print
    train_from_cache()
    
