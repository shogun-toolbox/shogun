require 'narray'
require 'modshogun'

# for debugging purposes...
#require 'rubygems'
#require "pry"

def LoadMatrix(filename, type = :numbers)
  case type
  when :numbers
    x = LoadMatrix.load_numbers(filename)
    pp x
  when :dna
    LoadMatrix.load_dna(filename)
  when :cubes
    LoadMatrix.load_cubes(filename)
  when :labels
    LoadMatrix.load_labels(filename)
  end
end

module LoadMatrix
  def load_numbers(filename)
    matrix = []
    File.open(filename) do |file|
      file.each_line do |line|
        ary = []
        line.split(" ").each{ |n| ary << n.to_f }
        matrix << ary
      end
    end
    matrix = NArray.to_na(matrix)
    matrix = matrix.transpose(1,0)
  end

  def load_dna(filename)
    puts 'loading dna'
    matrix=[]
    File.open(filename) do |file|
      file.each_line do |line|
        line.split("\n").each{|n| matrix << n }
      end
    end
    matrix
  end


  def load_cubes(filename)
    matrix=[]
    File.open(filename) do |file|
      file.each_line do |line|
        line.split("\n").each{|n| matrix << n }
      end
    end
    matrix
  end


  def load_labels(filename)
    vector = []
    File.open(filename) do |file|
      file.each_line do |line|
        vector << line.to_f
      end
    end
    vector = NArray.to_na(vector)
    # loading vectors is not yet implemented,
    # use a 1D array instead (for now)
    #vector = vector.reshape(1, vector.total)
  end
 extend self
end

def randn mean = 0.0, std_dev = 1.0
  Modshogun::Math.normal_random mean.to_f, std_dev.to_f
end

# some stuff added to the Modshogun:: namespace

Modshogun::Math.class_eval do
  def rand
    # do something later
  end
end



Numeric.class_eval do
  def sign
    return -1 if self < 0
    return 0 if self == 0
    return 1 if self > 0
  end
end

module ArrayHelpers
  def sign
    a = []
    self.each do |x|
      a << x.sign
    end
    a
  end

  def eql_items? other
    raise(ArgumentError, "Arrays dont' have the same number of elements") if self.size != other.size
    output = []
    self.each_with_index do |x, i|
      output[i] = x == other[i]
    end
    return output
  end

  # implement like numpy diag
  # http://docs.scipy.org/doc/numpy/reference/generated/numpy.diag.html?highlight=diag#numpy.diag
  def diag
  end
end

NArray.class_eval do
  include ArrayHelpers
end

Array.class_eval do
  include ArrayHelpers
end

# maybe include some of the following in the Modshogun:: namespace proper...

# yes this is a very mean & dirty mean alg...
# no checking of stuff, can give wonky errors, but hey!
# i'm being lazy & not doing cool metaprogramming type stuff for now...
def mean ary
  num_items = ary.size
  tot = ary.inject(0) do |sum, n|
    if n == true
      sum + 1
    elsif (n == false) or (n == nil)
      sum + 0 # yes it's a dummy
    else
      sum + n
    end
  end
  tot.to_f / num_items.to_f
end

# some array generating methods
def gen_pos_ones_vec num
  ary = []
  num.times do
    ary << 1
  end
  return ary
end

def gen_neg_ones_vec num
  ary = []
  num.times do
    ary << -1
  end
  return ary
end

def gen_ones_vec num
  ary = []
  num.times do
    ary << -1
  end
  num.times do
    ary << 1
  end
  return ary
end

# 2 high, num wide random arrays, concatenated together [] + []
# randn - dist, randn + dist
def gen_rand_ary num, dist=0
  ary = [[],[]]
  ary.map! do |p|
    p << ary_fill( num, dist ) + ary_fill( num, -dist )
    p.flatten!
  end
  return ary
end

def ary_fill num, dist=0
  ary = []
  num.times do
    ary << randn + dist
  end
  return ary
end
