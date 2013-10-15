#!/usr/bin/perl -I ../../../src/interfaces/perl_modular -I .
#TODO:PTZ121027 remove this hack by using use a Env{???}
use lib  qw(../../../src/interfaces/perl_modular);
use PDL;
use modshogun;
use generator qw(setup_tests get_fname blacklist get_test_mod run_test);

use Test::More;
use Data::Dumper;
use File::Slurp;
use Getopt::Long;
use Pod::Usage;

use Data::Dumper;
use Digest::MD5 qw(md5 md5_hex md5_base64);
use IO::String;
use IO::File;
use Data::Hexdumper qw(hexdump) ;
use File::Temp qw/tempfile tempdir mktemp/;
use File::Spec;
use Tie::IxHash;
#use MIME::Lite::HTML;
#use Algorithm::QuineMcCluskey::Util;
use Storable;
use Archive::Zip qw( :ERROR_CODES );


sub typecheck
{
    my ($a, $b) = @_;
    (my $rfa = ref($a)) =~ s/\(.*$//;
    (my $rfb = ref($b)) =~ s/\(.*$//;
    if($rfa =~ /Shogun.*Labels/ and $rfb =~ /Shogun.*Labels/) {
	return 1;
    }
    return($rfa eq $rfb);
}

#PTZ121006 there must be better ways
sub compare
{
    my($a, $b, $tolerance) = @_;
    if(not &typecheck($a, $b)) { return 0;}
    if(ref($a) =~ /PDL::ndarray/) {
	if($tolerance) {
	    return (max(abs($a - $b)) < $tolerance);
	} else {
	    return like($a,$b);
	}
    } elsif( &isinstance($a, modshogun::SGObject)) {
	return like(Dumper($a), Dumper($b));
    } elsif(ref($a) =~ qr'ARRAY') {
	if($#$a != $#$b) {return 0;}
	while((my ($obj1) = pop(@$a)) && (my ($obj2) = pop(@$b))) {
	    if( not &compare($obj1, $obj2, $tolerance)) {
		return 0;
	    }
	}
	return 1;
    }
    return $a <=> $b;
}

sub compare_dbg {
    my ($a, $b, $tolerance) = @_;
    if(not &compare_dbg_helper($a, $b, $tolerance)) {
#	import pdb;
#	pdb.set_trace()
    }
}

sub compare_dbg_helper
{
    my ($a, $b, $tolerance) = @_;
    my $rfa = ref($a);
    my $rfb = ref($b);
    if(not &typecheck($a, $b)) {
	printf( "Type mismatch (type(a)=%s vs type(b)=%s)", ref($a), ref($b));
	    return 0;
    }
    if(ref($a) =~ /PDL/) {
	if ($tolerance){
	    if (max(abs($a - $b)) < $tolerance) {
		return 1;
	    }else{
		print "PDL Array mismatch > max_tol";
		print $a-$b;
		return 0;
	    }
	}else{
	    if (&is_like($a, $b)){
		return 1;
	    }else{
		print "PDL Array mismatch";
		print $a-$b;
		return 0;
	    }
	}
    } elsif(ref($a) =~ /modshogun::SGObject/){
	if(&like(Dumper($a), Dumper($b))) {
	    return 1;
	}
	print("a=", Dumper($a));
	print("b=", Dumper($b));
	return 0;
    } elsif( $rfa =~ 'ARRAY') {
	if($#a != $#b) {
	    printf( "Length mismatch (len(a)=%d vs len(b)=%d)", $#a, $#b);
	    return 0;
	}
	while(my ($obj1, $obj2) = each &zip($a,$b)) {
	    if( not &compare_dbg($obj1, $obj2, $tolerance)) {
		return 0;
	    }
	}
	return 1;
    }

    if ($a == $b) {
	return 1;
    } else {
	print "a!=b";
	print "a=", $a;
	print "b=", $b;
	return 0;
    }
}
sub tester
{
    my ($tests, $cmp_method, $tolerance, $failures, $missing) = @_;
    foreach my $t (@$tests) {
	my $n = 0;
	my ($mod, $mod_name) = &get_test_mod($t);
	if($mod && 0) {
#TODO::PTZ121109 parameter_list nowhere to be seen... maybe loadfile_parameter
	    $n = $#{$mod->{parameter_list}};
	    unless($n >= 0) {next;}
	    if($@) {
		warn( "%-60s ERROR (%s)", $t, $@);
		next;
	    }
	}
	my $fname = "";
	foreach my $i (0..$n) {
	    $fname = &get_fname($mod_name, $i);
	    my $setting_str = sprintf("%s setting %d/%d", $t, $i+1, $n);
	    my $a = &run_test($mod, $mod_name, $i);
	    my $b = &read_file($fname);#slurp file...
	    if(&cmp_method($a, $b, $tolerance)) {
		if(not $failures and not $missing) {
		    printf("%-60s OK", $setting_str);
		} else {
		    if(not $missing) {
			printf("%-60s ERROR", $setting_str);
		    }
		}
		#PTZ12106todo ... use ok(...); or so...
		if($@) {
		    warn($setting_str, $@);
		    #except IOError, e:
		    if(not $failures) {
			warn( "%-60s NO TEST", $setting_str);
		    }
		    #except Exception, e:
		    if (not $missing) {
			printf("%-60s EXCEPTION %s", $setting_str, $e);
		    }
		}
	    }
	}
    }
}


my $verbose = '';   # option variable with default value (false)
my $all = '';       # option variable with default value (false)
my $man = 0;
my $help = 0;

my $debug = false;
my $failures = false;
my $tolerance = false;
my $missing = false;
my $cmp_method;


GetOptions ('verbose' => \$verbose, 'all' => \$all,
	    , 'debug!' => \$debug
	    , 'help|?' => \$help, man => \$man
	    , 'failures!' => \$failures
	    , 'tolerance!' => \$tolerance
	    , 'missing!' =>\$missing
    ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

if($debug) {
    $cmp_method= \&compare_dbg;
} else {
    $cmp_method=\&compare;
}
my $tests = &setup_tests(\@ARGV);
&tester($tests, $cmp_method, $tolerance, $failures, $missing);

__END__


=head1 NAME

    tester.pl - Using Getopt::Long and Pod::Usage

 =head1 SYNOPSIS

    tester.pl [options]  [<file1> <file2> ...]

    Options:
        --help          brief help message
        --man           full documentation
	--debug		detailed debug output of objects that don't match
	--failures	show only failures
	--missing	show only missing tests
	--tolerance	tolerance used to estimate accuracy

=head1 OPTIONS

=over 8

=item B<--help>

    Print a brief help message and exits.

=item B<--man>

    Prints the manual page and exits.

=item B<--debug>

    detailed debug output of objects that don't match

=item B<--failures>
    show only failures

=item B<--missing>

    show only missing tests

=item B<--tolerance>

    tolerance used to estimate accuracy

=back

=head1 DESCRIPTION

           B<This program> will read the given input file(s) and do something
           useful with the contents thereof.


=cut


=pod


 #from Makefile.PL

  check-local: Makefile.perl
	$(MAKE) -f $< test

    done

=cut
