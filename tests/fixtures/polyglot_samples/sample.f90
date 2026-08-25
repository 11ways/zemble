module store
  implicit none
  type :: point
    integer :: x
  end type point
contains
  function area(p) result(r)
    type(point), intent(in) :: p
    integer :: r
    r = helper(p%x)
  end function area
  subroutine run(n)
    integer :: n
    call other(n)
  end subroutine run
  integer function helper(n)
    integer :: n
    helper = n + 1
  end function helper
end module store
