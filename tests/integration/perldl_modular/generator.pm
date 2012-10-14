#!/usr/bin/perl
package generator;

use IO::File;
use File::Basename;
use File::Find::Rule;
use File::Slurp;
use File::Spec;
use File::pushd;
use base qw(Exporter);

our %EXPORT_TAGS =
(
 #FIELDS => [ @EXPORT_OK, @EXPORT ]
 all => [qw(setup_tests get_fname blacklist get_test_mod run_test)]
);
#Exporter::export_tags(qw/all/);
Exporter::export_ok_tags(qw/all/);

our $example_dir = '../../../examples/undocumented/perl_modular';
our $test_dir = '../../regression/tests';
our @blacklist = ("classifier_libsvm_minimal_modular.t",
		"kernel_combined_modular.t",
		"kernel_distance_modular.t",
	      "distribution_hmm_modular.t");

sub get_fname
{
    my ($mod_name, $i) = @_;
    return File::Spec->catfile($test_dir, $mod_name . $i . '.txt');
}

sub setup_tests($) {
    my ($tests) = @_;
    my $edir = pushd( $example_dir );
    if($#$tests < 0) {
	$tests = File::Find::Rule->file()
	    ->name( '*.t' )
	    ->in( '.' );
    } else {
	$tests = [ map(basename($_), @$tests) ];
    }
    #sys.path.insert(0, '.')
    return $tests;
}

sub check_for_function($)
{
    my ($fname) = @_;
    my $fh = IO::File->new($fname, "r");
    if (defined $fh) {
	while(my $l = <$fh>) {
	    if($l =~ /^\s*sub /) {
		return true;
	    }
	}
    }
    return false;
}
sub get_test_mod
{
    my ($t) = @_;
    if (($t =~ /.t$/)
	and not ($t =~ /^\./)
	and !(grep($t, @blacklist)))
    {
	my $mod_name = $t =~ s/\..*$//;
	if(not &check_for_function($t)) {	    
	    warn("ERROR (no function)");
	}
        #return __import__(mod_name), mod_name
	return(ref($mod_name), $mod_name);
    }
}

sub run_test
{
    my ($mod, $mod_name, $i) = @_;
    my $fname = &get_fname($mod_name, $i);
    my $par = $mod->[$i];
    my $a =  &getattr($mod, $mod_name)->($par);
    return $a;
}


sub generator
{
    my ($tests) = @_;
    foreach  my $t (@$tests) {
	my ($mod, $mod_name) = &get_test_mod($t);
	unless( @$mod ) { next; }
	if($@) {
	    warn("%-60s", $mod_name);	    
	    next;
	}
	my $fname = "";

	printf("%-60s", $mod_name);
	#print "%+60s" % "...",
	foreach my $i (0..$#{$mod}) {	    
	    $fname = &get_fname($mod_name, $i);
	    my $a = &run_test($mod, $mod_name, $i);
	    write_file($fname, $a);
	    if(@_) {
		warn( "ERROR generating '%s' using '%s'" , $fname, $t);
		next;
	    }
	    print "OK";
	}
    }
}

true;
__END__
=head2 SYNOPSYS

  my $tests = &setup_tests(\@ARGV);
  &generator($tests);

=cut


import os
import sys
import pickle

example_dir = '../../../examples/undocumented/perl_modular'
test_dir = '../../regression/tests'
blacklist = ("__init__.py", "classifier_libsvm_minimal_modular.t",
		"kernel_combined_modular.t",
		"kernel_distance_modular.t",
		"distribution_hmm_modular.t")

def get_fname(mod_name, i):
	return os.path.join(test_dir, mod_name + str(i) + '.txt')

def setup_tests(tests=[]):
	os.chdir(example_dir)

	if not len(tests):
		tests =  os.listdir(".")
		tests.sort()
	else:
		tests = [ os.path.basename(t) for t in tests ]

	sys.path.insert(0, '.')
	return tests

def check_for_function(fname):
	for l in file(fname).readlines():
		if l.startswith("def "):
			return True
	return False

def get_test_mod(t):
	if t.endswith(".t") and not t.startswith('.') and t not in blacklist:
		mod_name = t[:-3]

		if not check_for_function(t):
			raise Exception("ERROR (no function)")

		return __import__(mod_name), mod_name

def run_test(mod, mod_name, i):
	fname = get_fname(mod_name, i)
	par=mod.parameter_list[i]
	a =  getattr(mod, mod_name)(*par)
	return a

def generator(tests):
	for t in tests:
		try:
			mod, mod_name = get_test_mod(t)
		except TypeError:
			continue
		except Exception, e:
			print "%-60s" % mod_name,
			print e
			continue
		fname = ""

		print "%-60s" % mod_name,
		#print "%+60s" % "...",
		try:
			for i in xrange(len(mod.parameter_list)):
				fname = get_fname(mod_name, i)
				a = run_test(mod, mod_name, i)
				pickle.dump(a,file(fname, "w"))
			print "OK"
		except Exception, e:
			print "ERROR generating '%s' using '%s'" % (fname,t)
			print e
			continue

if __name__=='__main__':
	from optparse import OptionParser
	op=OptionParser()
	op.set_usage("[<file1> <file2> ...]")
	(opts, args)=op.parse_args()
	tests = setup_tests(args)
	generator(tests)
