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
#TODO::PTZ121109 relpath is quite wrong
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
#TODO::PTZ121109
    my ($fname) = @_;
    my $fh = IO::File->new($fname, "r");
    if (defined $fh) {
	while(my $l = <$fh>) {
	    if($l =~ /^\s*sub /) {
		return 1;
	    }
	}
    }
    return 0;
}
sub get_test_mod
{
    my ($t) = @_;
    if (($t =~ /.t$/)
	and not ($t =~ /^\./)
	and !(grep(/$t/, @blacklist)))
    {
	(my $mod_name = $t) =~ s/\..*$//;
	my $mod = $mod_name;
	$mod =~ s#.*/##g;
	$mod =~ s#_.*##g;
	#TODO::PTZ121109 $t is path less...
	#if(not &check_for_function($t)) {
	#TODO::PTZ121109   warn("ERROR (no function)");
	#}
	$mod = eval( 'modshogun::' . $mod)->new();
	#TODO::PTZ121109 something for $mod->{parameter_list}
	return($mod, $mod_name);
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

1;
__END__

=head2 SYNOPSYS

  my $tests = &setup_tests(\@ARGV);
  &generator($tests);

=cut
