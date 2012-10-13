#!/usr/bin/perl

use lib qw(. /usr/src/shogun/src/interfaces/perldl_modular /usr/src/shogun/src/shogun);
use PDL;
use PDL::Char;

use IO::File;

import Devel::Trace 'trace';

use kernel;
use distance;
use classifier;
use clustering;
use distribution;
use regression;
use preprocessor;

#...
our @SUPPORTED=('kernel', 'distance', 'classifier', 'clustering', 'distribution',
		'regression', 'preprocessor');

#use Shogun qw(Math_init_random);
use modshogun;

sub _get_name_fun($)
{
    my $fnam = shift;
    my $module;
    if(my ($supported) = grep($fnam =~ /$_/, @SUPPORTED)) {
	$module = $supported;
    }
    unless($module) {
	printf('Module required for %s not supported yet!', $fnam);
	return undef;
    }
    return $module . '::test';
}

=head3 _test_mfile

 a simple parser from m files to perl structures

=cut

sub _test_mfile {
    my $fnam = shift;    
    my $mfile = IO::File->new($fnam, 'r') or return false;	
    my %indata = ();

    my $name_fun = &_get_name_fun($fnam);
    unless($name_fun) {
	return false;
    }
    while(my $line = <$mfile>) {
	$line =~ s/[\s]//g;
	$line =~ s/;$//;
	(my $param = $line) =~ s/=.*//;

	if($param eq 'name') {
	    $indata{$param} = $line =~ m/.*='(.*)'/;
	} elsif ($param eq 'kernel_symdata' or $param eq 'kernel_data') {
	    $indata{$param} = &_read_matrix($line);
	} elsif ($param =~ /^kernel_matrix/ or 
		 $param =~ /^distance_matrix/) {
	    $indata{$param} = &_read_matrix($line);
	} elsif ($param =~ /data_train/ or $param =~ /data_test/) {
	    # data_{train,test} might be prepended by 'subkernelX_'
	    $indata{$param} = &_read_matrix($line);
	} elsif ($param eq 'classifier_alphas' or $param eq 'classifier_support_vectors') {
	    ($indata{$param}) = $line =~ m/=(.*)$/;
	    unless($indata{$param}) {
		# might be MultiClass SVM and hence matrix
		$indata{$param} = &_read_matrix($line);
	    }
	} elsif($param eq 'clustering_centers' or $param eq 'clustering_pairs') {
	    $indata{$param} = &_read_matrix($line); 
	} else {
	    unless($line =~ /'/) {
		my $str_line;
		if(($str_line) = $line =~ m/=\[(.*)\]$/g) {
		    $indata{$param} = &pdl(eval($str_line));
		} elsif(($str_line) = $line =~ m/=\{(.*)\}$/g) {
		    $indata{$param} = PDL::Char->new(eval($str_line));
		} else {
		    ($indata{$param}) = $line =~ m/=(.*)$/;
		}
	    } else {
		($indata{$param}) = $line =~ m/='(.*)'/;
	    }
	}
    }
    $mfile->close();
    my $fun = *{$name_fun};
    my $random = modshogun::Math->new();
    # seed random to constant value used at data file's creation
    &modshogun::Math::init_random($indata{'init_random'});
    #is(&modshogun::Math::get_seed(), $indata{'init_random'});
    #$random->seed($indata{'init_random'});
    #= &modshogun::Math::random()
    return $fun->(\%indata);
}

sub _stringlike {
  return 1 unless defined $_[0] && ref $_[0];
  return 1 if (blessed $_[0]) &&
    eval {
      (("$_[0]" eq "$_[0]") && (($_[0] cmp $_[0])==0))
    };
  return;
}
sub _numberlike {
  return 1, unless defined $_[0];

  # L<perlfunc> manpage notes that NaN != NaN, so we can verify that
  # numeric conversion function works properly along with the
  # comparison operator.
  no warnings;

  return 1 if ((!ref $_[0]) || blessed($_[0])) &&
    eval {
      ((0+$_[0]) == (0+$_[0])) && (($_[0] <=> $_[0])==0)
    } && ($_[0] =~ /\d/);
  return;
}
sub _read_matrix {
    my $line = shift;
    my $is_char = 0;
    my ($str_line) = $line =~ m/\[(.*)\]/g;
    unless($str_line) {
	($str_line) = $line =~ m/\{(.*)\}/g;
	$is_char = 1;
    }
    my @lines = split(/;/, $str_line);
    my @lis2d;

    foreach my $x (@lines) {
	my @lis;
	foreach my $y (split(/,/, $x)) {
	    if($y =~ /'/) {
		$is_char = 1;	
	    }
	    $y =~ s/'//g;
	    push(@lis, $y);
	    #$is_char ||= (&_stringlike($y) && !&_numberlike($y));
	}
	push(@lis2d, \@lis);
    }
    if($is_char) {
	my $m = PDL::Char->new(\@lis2d);
	return $m;
    }
    PDL->new(\@lis2d);
}

my $res = 1;
foreach my $filename (@ARGV) {
    if($filename =~ /\.m$/) {
	$res &&= &_test_mfile($filename);
    }
}

$res;

__END__

=head1 

Test one data file

=cut

from numpy import *
import sys

import kernel
import distance
import classifier
import clustering
import distribution
import regression
import preprocessor
from shogun.Library import Math_init_random

SUPPORTED=['kernel', 'distance', 'classifier', 'clustering', 'distribution',
	'regression', 'preprocessor']

def _get_name_fun (fnam):
	module=None

	for supported in SUPPORTED:
		if fnam.find(supported)>-1:
			module=supported
			break

	if module is None:
		print 'Module required for %s not supported yet!' % fnam
		return None

	return module+'.test'

def _test_mfile (fnam):
	try:
		mfile=open(fnam, mode='r')
	except IOError, e:
		print e
		return False
	
	indata={}

	name_fun=_get_name_fun(fnam)
	if name_fun is None:
		return False

	for line in mfile:
		line=line.strip(" \t\n;")
		param = line.split('=')[0].strip()

		if param=='name':
			name=line.split('=')[1].strip().split("'")[1]
			indata[param]=name
		elif param=='kernel_symdata' or param=='kernel_data':
			indata[param]=_read_matrix(line)
		elif param.startswith('kernel_matrix') or \
			param.startswith('distance_matrix'):
			indata[param]=_read_matrix(line)
		elif param.find('data_train')>-1 or param.find('data_test')>-1:
			# data_{train,test} might be prepended by 'subkernelX_'
			indata[param]=_read_matrix(line)
		elif param=='classifier_alphas' or param=='classifier_support_vectors':
			try:
				indata[param]=eval(line.split('=')[1])
			except SyntaxError: # might be MultiClass SVM and hence matrix
				indata[param]=_read_matrix(line)
		elif param=='clustering_centers' or param=='clustering_pairs':
			indata[param]=_read_matrix(line)
		else:
			if (line.find("'")==-1):
				indata[param]=eval(line.split('=')[1])
			else:
				indata[param]=line.split('=')[1].strip().split("'")[1]

	mfile.close()
	fun=eval(name_fun)

	# seed random to constant value used at data file's creation
	Math_init_random(indata['init_random'])
	random.seed(indata['init_random'])

	return fun(indata)

def _read_matrix (line):
	try:
		str_line=(line.split('[')[1]).split(']')[0]
	except IndexError:
		str_line=(line.split('{')[1]).split('}')[0]

	lines=str_line.split(';')
	lis2d=list()

	for x in lines:
		lis=list()
		for y in x.split(','):
			y=y.replace("'","").strip()
			if(y.isalpha()):
				lis.append(y)
			else:
				if y.find('.')!=-1:
					lis.append(float(y))
				else:
					try:
						lis.append(int(y))
					except ValueError: # not int, RAWDNA?
						lis.append(y)

		lis2d.append(lis)

	return array(lis2d)

for filename in sys.argv:
	if (filename.endswith('.m')):
		res=_test_mfile(filename)
		if res:
			sys.exit(0)
		else:
			sys.exit(1)
