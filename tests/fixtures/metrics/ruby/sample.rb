# A comment line.
def seq(a)
  b = a + 1
  c = b * 2
  c
end

def nested(xs, a, b)
  for x in xs
    if a
      if b
        work(x)
      end
    end
  end
end

def branches(a)
  if a > 1
    1
  elsif a > 0
    2
  else
    3
  end
end

def booleans(a, b, c)
  if a && b && c
    work(1)
  end
  if a && b || c
    work(2)
  end
end

def ternary(a)
  a ? 1 : 2
end

def try_catch(a)
  begin
    work(1)
  rescue StandardError
    if a
      work(2)
    end
  end
end

def fact(n)
  if n <= 1
    return 1
  end
  n * fact(n - 1)
end

def lambda_in_loop(xs, a)
  for x in xs
    run(-> {
      if a
        work(x)
      end
    })
  end
end

def work(x)
end

def run(r)
end
