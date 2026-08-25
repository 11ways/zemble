module Store.Point (area) where
import Data.List
data Shape = Circle Double | Square Double deriving (Show)
class Area a where
  area :: a -> Double
instance Area Shape where
  area (Circle r) = helper r
  area (Square s) = s
newtype Wrapper = Wrapper Int
type Alias = Int
helper :: Double -> Double
helper n = n * 2
