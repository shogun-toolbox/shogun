require 'modshogun'

# for random numbers:
def randn
  Modshogun::Math.randn_double
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
end

NArray.class_eval do
  include ArrayHelpers
end

Array.class_eval do
  include ArrayHelpers
end

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
def gen_ones_vec
  ary = []
  @num.times do
    ary << -1
  end
  @num.times do
    ary << 1
  end
  return ary
end

# 2 high, num wide random arrays, concatenated together [] + []
# randn - dist, randn + dist
def gen_rand_ary
  ary = [[],[]]
  ary.each do |p|
    p << ary_fill( @dist ) + ary_fill( -@dist )
    p.flatten!
  end
  return ary
end

def ary_fill dist
  ary = []
  @num.times do
    ary << randn + dist
  end
  return ary
end