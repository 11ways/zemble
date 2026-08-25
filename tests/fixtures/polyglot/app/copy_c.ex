defmodule Copy do
  def render(rows) do
    rows
    |> Enum.map(fn row -> row |> Enum.map(&String.trim/1) |> Enum.join(" | ") end)
    |> Enum.join("\n")
  end

  defp hidden(x), do: x + 1
end
