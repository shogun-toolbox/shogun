#!/usr/bin/perl -w
#
use File::Temp;

use FileHandle;
use File::Spec			qw(catfile rel2abs tmpdir);
use File::Basename		qw[dirname basename];
use File::Path			qw(remove_tree make_path);
use File::Find::Rule;
use File::Find::Rule::VCS;
use File::Copy;
use File::Copy::Recursive qw(fcopy rcopy dircopy fmove rmove dirmove);
use IO::File;

use strict;
use Storable;

my %s = (
    dir  => File::Spec->catfile('/usr/src/shogun')
    );
sub _duplicate_py_perl($$) {
    my $s = shift;
    my $path = shift || $s->{dir};
    my @fs_i = File::Find::Rule->ignore_vcs->in($path);
    my $r;
    my $tree_done = '';
    while(my $fh_i_n = shift(@fs_i)) {
	my $fh_i_bn = basename($fh_i_n);
	if($fh_i_bn =~ /python/i) {
	    (my $fh_i_p = $fh_i_n) =~ s/ython/erl/g;
	    $r = rcopy($fh_i_n, $fh_i_p);
	    $r = &_duplicate_py_perl($s, $fh_i_p);
	    next; #for files or directories
	} elsif($fh_i_bn =~ /\.py$/i) {
	    (my $fh_i_p = $fh_i_n) =~ s/\.py$/\.pl/g;
	    $r = rcopy($fh_i_n , $fh_i_p);
	    $r = &_duplicate_py_perl($s, $fh_i_p);	    
	}
	if($fh_i_n =~ /shogun\/.*(python.*\.pl|perl.*\.py)$/i) {
	    unlink($fh_i_n);
	    next;
	}
	if($fh_i_n =~ /python|\.py|duplicate_py_perl.pl/i) {
#uhmm, there again this is not straight forward, do we authorise changes in perl files?
	    next;
	}
	if(-f $fh_i_n) {
	    my @i_l_a;
	    my $e = \%{$s->{$fh_i_n}};
	    foreach my $k (qw/ifdef ifdef_endif endif fi if if_fi lbrack rbrack l_r func in_func python PYTHON/) {
		$e->{$k} = 0;
	    }	    
	    my $fh_i_e = IO::File->new($fh_i_n, 'r');
	    my $fh_i_p = File::Temp->new
		(TEMPLATE => basename($fh_i_n) . '_XXXX'
		 , DIR => dirname($fh_i_n)
		 #, SUFFIX => '.' . $g_sfx
		 , UNLINK => 1
		) or do {
		    croak $!;
	    };
	    while(my $l = $fh_i_e->getline()) {
		my $i_l = $l;

#check for embrication.: if(and) then else(or), ifdef endif,  functional () {...} ,
#and others!!!
# push line number, and duplicate the whole, only in one depth also

		$l =~ /^\s*#\s*if\s*def(\s|ined)/ && do {
		    $e->{ifdef} += 1;
		    $e->{ifdef_endif} = 1;
		};
		$l =~ /^\s*#\s*endif/ && do {
		    $e->{endif} += 1;
		    $e->{ifdef_endif} = -1;
		};
		$l =~ /^\sif\s+/ && do {
		    $e->{'if'} += 1;
		    $e->{if_fi} = 1;
		};
		$l =~ /^\sfi\s*/ && do {
		    $e->{fi} += 1;
		    $e->{if_fi} = -1;
		};
		$l =~ /^{\s*$/ && do {
		    $e->{lbrack} += 1;
		    $e->{l_r} = 1;
		};
		$l =~ /^}\s*$/ && do {
		    $e->{rbrack} += 1;
		    $e->{l_r} = -1;
		};
		$l =~ /\w\(.*\)\s*$/ && do {
		    $e->{func} += 1;
		};

		$l =~ /[Pp]ython/ && do {
		    $i_l =~ s/ython/erl/g;
		    $e->{python} += 1;
		    $e->{in_func} += 1;
		};
		$l =~ /PYTHON/ && do {
		    $i_l =~ s/PYTHON/PERL/g;
		    $e->{PYTHON} += 1;
		    $e->{in_func} += 1;
		};
		
		#estimate end of embrication if_fi & l_r inside macro ifdef_endif !
		if(
		    (($e->{ifdef_endif} < 0) && ($e->{ifdef} == $e->{endif}))
		    || (
			($e->{ifdef} ==  $e->{endif})
			&& (#both l_r and  if_fi
			    (($e->{if_fi} < 0) || ($e->{l_r} < 0))
			    && ($e->{'if'} == $e->{fi})
			    && ($e->{lbrack} == $e->{rbrack})
			)
			)			
		    )
#still prob of macro inside if then else! too tired do it manually
		{
		    #string change occured?
		    if($e->{in_func}) {
			#closure we would need apend the bunch of lines.
			$i_l .= "\n";
			$i_l .= join("\n", pop(@i_l_a));
			$e->{in_func} = 0;
		    } else {
			#never mind
			pop(@i_l_a);
		    }	    
		}
		if(($e->{if_fi} < 0) && ($e->{'if'} ==  $e->{fi})) {
		    #string change occured?
		    if($e->{in_func}) {
			#closure we would need apend the bunch of lines.
			$i_l .= "\n";
			$i_l .= join("\n", pop(@i_l_a));
			$e->{in_func} = 0;
		    } else {
			#never mind
			pop(@i_l_a);
		    }	    
		}
		$e->{ifdef_endif} = 0;
		$e->{if_fi} = 0;
		$e->{l_r} = 0;

		#estimate start and continum of embrication
		if(
		    ($e->{ifdef} >  $e->{endif})
		    || (($e->{ifdef} == $e->{endif})
			&& (($e->{'if'} >  $e->{fi})
			    || ($e->{lbrack} >  $e->{rbrack})
			)
		    )
		    )
		{
		    unless(@i_l_a) {
			#first if,then else
			push(@i_l_a, []);
			$e->{in_func} = ($l =~ m/python/i);		
		    }
		    #duplicate
		    push(@{$i_l_a[-1]}, "$i_l");
		    $i_l = $l;
		} elsif($l ne $i_l) {
		    $fh_i_p->print($l);
		    #and save change...
		}
		$fh_i_p->print($i_l);
	    }	    
#flush errors
#close
	    if($e->{PYTHON} || $e->{python}) {
		$fh_i_p->close;
		$fh_i_e->close;
		fcopy($fh_i_p->filename,  $fh_i_n);
	    }
	}
    }		       
}


my $r = &_duplicate_py_perl(\%s);

#local $Storable::Deparse = 1;
store(\%s, File::Spec->catfile($s{dir}, '.duped_py_pl.pb'));


__END__

=head1 NAME

    duplicate_py_perl.pl

=head1 AUTHOR and RECOMMENDATIONS

    (c) 23/09/2012 by <ptizoom@gmail.com>.
    use freely but at your own risks,
    as this script may remove important 
    data in your /usr/src/shogun directory.

=head3 SYNOPSIS

=item

    cd /usr/src/shogun
    git checkout -t -b perl_swig_120921
    ./contrib/duplicate_py_perl.pl

=item

and to revert changes:
    git reset
    git ls-files --others | xargs rm -v

=cut
