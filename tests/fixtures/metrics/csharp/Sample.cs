// A comment line.
class Sample
{
    int Seq(int a)
    {
        int b = a + 1;
        int c = b * 2;
        return c;
    }

    void Nested(int[] xs, bool a, bool b)
    {
        foreach (var x in xs)
        {
            if (a)
            {
                if (b)
                {
                    Work(x);
                }
            }
        }
    }

    int Branches(int a)
    {
        if (a > 1)
        {
            return 1;
        }
        else if (a > 0)
        {
            return 2;
        }
        else
        {
            return 3;
        }
    }

    void Booleans(bool a, bool b, bool c)
    {
        if (a && b && c)
        {
            Work(1);
        }
        if (a && b || c)
        {
            Work(2);
        }
    }

    int Ternary(bool a)
    {
        return a ? 1 : 2;
    }

    void TryCatch(bool a)
    {
        try
        {
            Work(1);
        }
        catch (System.Exception e)
        {
            if (a)
            {
                Work(2);
            }
        }
    }

    int Fact(int n)
    {
        if (n <= 1)
        {
            return 1;
        }
        return n * Fact(n - 1);
    }

    void LambdaInLoop(int[] xs, bool a)
    {
        foreach (var x in xs)
        {
            Run(() =>
            {
                if (a)
                {
                    Work(x);
                }
            });
        }
    }

    void Work(int x) {}

    void Run(System.Action r) {}
}
