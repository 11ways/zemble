function! Area(x) abort
  return s:helper(a:x)
endfunction
function s:helper(n)
  return a:n + 1
endfunction
let g:limit = 3
command! Run call Area(1)
