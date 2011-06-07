#!/usr/bin/env ruby

# some tests to see if the shogun libraries are loading properly

require 'test/unit'
require './metaid' # i know, i should use require_relative, but this may be run my 1.8.x-ers

class TestLibs < Test::Unit::TestCase

  def setup
    @libs = %w{ Classifier Distance Evaluation Kernel Preprocessor Structure Clustering Distribution Library Regression }
  end
  
  # loads each module iterativly, to see if it loads
  def test_load
    # randomize the order of the libs & print them so we don't get weird X depends on Y loading errors
    puts "Shuffling the order of the libraries to be loaded: " + @libs.shuffle!.to_s
    @libs.each do |lib|
      assert load_lib( lib ), "ERROR!!!!! #{lib} didn't load!!"
    end
  end

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