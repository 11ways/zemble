package Store::Point;
use strict;
sub new { my ($class, %args) = @_; return bless {%args}, $class; }
sub area { my $self = shift; return helper($self->{x}); }
sub helper { my $n = shift; return $n + 1; }
1;
