#!/usr/bin/env ruby

# serialhex's awesome quick-and-dirty python-to-ruby trancekoder of indomnitable
# awesomeness!!
#
# use: for quickly trancekoding python source into ruby source...
# in the context of shogun anyway, it's probably not going to work too well with
# projects of any *real* size
# (plus, i'm hardcoding a number of things, so tough noogies!)
#
# anyway, this should be pretty interesting, as i imagine there will be a *lot*
# of regexes & stuff i haven't played with much before... so lets see how bad i
# break things!!!

require 'rubygems'
require 'pry'

class Trancekoder

  def initialize(filename)
    @file_lines = File.readlines(filename)
    @filename = filename.partition('../python_modular/')[2].sub /py/, 'rb'
  end

  attr_reader :filename

  def trancekode
    strip_python_loads #fin
    convert_data_loads # fin
    convert_methods # fin
    add_header # fin
    if_script # fin
    clean_up
  end

  def if_script
    @file_lines.each do |line|
      line.sub! /if __name__=='__main__':/, "if __FILE__ == $0"
    end
    @file_lines.last << "\nend\n"
  end

  def strip_python_loads
    @file_lines.each_index do |line_num|
      if @file_lines[line_num] =~ /\s*from[ A-Za-z0-9\_.]*import\s*\w*/
        @file_lines[line_num] = ''
      end
    end
  end

  def convert_data_loads
    @file_lines.each_index do |line_num|
      # yes i know there is an easier/faster way to do this...
      # shut up while i figure the rest of this out!!!
      if @file_lines[line_num] =~ /lm=LoadMatrix\(\)/
        @file_lines[line_num] = ''
      end
      @file_lines[line_num].sub! /lm\./, "LoadMatrix."
    end
  end

  def add_header
    ary = ["require 'narray'\n", "require 'modshogun'\n", "require 'load'\n", "require 'pp'\n"]
    @file_lines = ary + @file_lines
  end

  def convert_methods
    flag = false
    @file_lines.each_with_index do |line, line_num|
      if (flag == true) and (line =~ /^\S/)
        @file_lines[line_num] = "\nend\n" + line
        flag = false
      end
      if line =~ /def/
        flag = true
        line.sub! /\s+\(/, '('
        line.sub! /\):/, ')'
      end
    end
  end

  def clean_up
    @file_lines[0] = "# this was trancekoded by the awesome trancekoder\n" + @file_lines[0]
  end

  def to_s
    @file_lines.to_s
  end
end


## the goods at work
if __FILE__ == $0
  files = Dir.glob '../python_modular/*.py'
  files.each do |file|
    kode = Trancekoder.new file
    kode.trancekode
    File.open(kode.filename, 'w') do |f|
      f = kode.to_s
    end unless File.exists? kode.filename
  end
end