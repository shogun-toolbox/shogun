#!/usr/bin/env ruby

# some tests to see if the shogun libraries are loading properly

# instead of...
#require 'test/unit'
# use...
require './test_mod'

class TestLibs < Test::Unit::TestCase

  def setup
    @libs = %w{ Classifier Distance Evaluation Kernel Preprocessor Structure Clustering Distribution Library Regression }
  end
  
  # loads each module iterativly, to see if it loads
  must "load modules" do
    # randomize the order of the libs & print them so we don't get weird X depends on Y loading errors
    puts "Shuffling the order of the libraries to be loaded: " + @libs.shuffle!.to_s
    @libs.each do |lib|
      assert load_lib( lib ), "ERROR!!!!! #{lib} didn't load!!"
    end
  end

  # helper method of awsumness!!
  def load_lib lib
    failures = 0
    begin
      return require lib
    rescue LoadError
      # cheap hack to load the modules relativly
      lib = "../../src/ruby_modular/" + lib
      failures += 1
      retry if failures <= 1
      return false
    end
  end

end