defmodule Store.Point do
  @moduledoc "doc"
  use GenServer
  import Enum
  defstruct x: 0
  def area(%{x: x}), do: helper(x)
  def run(n) do
    Helper.compute(n)
    area(n)
  end
  defp helper(n), do: n
  defmacro m(x), do: x
end
