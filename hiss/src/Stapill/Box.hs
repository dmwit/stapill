-- | Finite rectangular pictures with a movable bottom-left corner.
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE BlockArguments #-}
module Stapill.Box
	( Box
	, topRight
	, at
	, empty
	, full
	, singleton
	, fromMap
	, fromList
	, toMap
	, overlay
	, hcat
	, vcat
	, Position (..)
	, inside
	) where

import Control.Applicative (Alternative)
import Control.Monad (guard)
import Data.Default (Default, def)
import Data.Function (on)
import Data.Map (Map)
import qualified Data.Map as M
import Data.Ord (comparing)
import Dr.Mario.Model (Position (..))

-- The representation satisfies these invariants:
--
-- * `all (inside (topRight_ box)) (M.keys (at_ box def))`
-- * `at_ box p = M.mapKeys (p .+) (at_ box def)`

-- | A rectangle that may have some @a@s in it at various positions. The
-- rectangle has bottom-left corner @Position 0 0@ and 'topRight' is strictly
-- above and to the right of the rectangle.
data Box a = Box
	{ topRight_ :: !Position
	, at_ :: Position -> Map Position a
	} deriving Functor

topRight :: Box a -> Position
topRight = topRight_

at :: Box a -> Position -> Map Position a
at = at_

instance Foldable Box where foldMap f = foldMap (foldMap f) . flatten
instance Traversable Box where
	-- ick. but I don't see a way around flattening
	traverse f box = inflate <$> traverse (traverse f) (flatten box)

instance Eq a => Eq (Box a) where (==) = (==) `on` flatten
instance Ord a => Ord (Box a) where compare = comparing flatten

instance Show a => Show (Box a) where
	showsPrec precedence (flatten -> (tr, cells)) = showParen (precedence > 10) $
		showString "fromList " . shows tr . showChar ' ' . shows (M.toList cells)

instance Semigroup (Box a) where (<>) = overlay
instance Monoid (Box a) where mempty = def
instance Default (Box a) where def = Box def def

flatten :: Box a -> (Position, Map Position a)
flatten box = (topRight_ box, at_ box def)

-- Not exported; unlike 'fromMap' below this does not perform any validation.
inflate :: (Position, Map Position a) -> Box a
inflate (tr, cells) = Box tr (\bl -> M.mapKeysMonotonic (bl .+) cells)

-- | A picture that's fully transparent, but may have a nontrivial size.
empty :: Position -> Box a
empty = flip Box def

-- | A picture that has the same content everywhere.
full :: Position -> a -> Box a
full tr a = Box tr \bl -> M.fromAscList [(bl .+ Position dx dy, a) | dx <- [0..x tr - 1], dy <- [0..y tr - 1]]

-- | A one-cell picture.
singleton :: a -> Box a
singleton = Box (Position 1 1) . flip M.singleton

-- | Construct a box after checking its rectangle and all stored keys.
fromMap :: Alternative f => Position -> Map Position a -> f (Box a)
fromMap extent cells = inflate (extent, cells) <$ guard
	(belowLeft def extent && all (inside extent) (M.keys cells))

-- | List counterpart to 'fromMap'.  Later duplicate keys replace earlier
-- ones, as with 'Data.Map.fromList'.
fromList :: Alternative f => Position -> [(Position, a)] -> f (Box a)
fromList extent = fromMap extent . M.fromList

-- | Return the cells in canonical bottom-left coordinates.
toMap :: Box a -> Map Position a
toMap = snd . flatten

data CoordinateMerge = CoordinateMerge { mergeTopRightCoordinate, mergeBottomLeftCoordinate :: Int -> Int -> Int }

aside, coincide :: CoordinateMerge
aside = CoordinateMerge (+) (+)
coincide = CoordinateMerge max const

mergeTopRight :: CoordinateMerge -> CoordinateMerge -> Position -> Position -> Position
mergeTopRight = mergePosition `on` mergeTopRightCoordinate

mergeBottomLeft :: CoordinateMerge -> CoordinateMerge -> Position -> Position -> Position
mergeBottomLeft = mergePosition `on` mergeBottomLeftCoordinate

overlay, hcat, vcat :: Box a -> Box a -> Box a
overlay = mergeBox coincide coincide
hcat = mergeBox aside coincide
vcat = flip (mergeBox coincide aside)
-- could kittyCorner = mergeBox aside aside, but from an interface perspective
-- there's really two sensible directions that things could be kitty-corner to
-- each other, so let's not pretend one is better than the other

-- The first argument is for merging 'x' coordinates, the second for 'y'
-- coordinates.
mergeBox
	:: CoordinateMerge
	-> CoordinateMerge
	-> Box a -> Box a -> Box a
mergeBox mergeX mergeY box box' = Box
	{ topRight_ = (mergeTopRight mergeX mergeY `on` topRight_) box box'
	, at_ = \bl -> M.union (box `at_` bl) (box' `at_` mergeBottomLeft mergeX mergeY bl (topRight_ box))
	}

mergePosition :: (Int -> Int -> Int) -> (Int -> Int -> Int) -> Position -> Position -> Position
mergePosition mergeX mergeY pos pos' = Position
	{ x = (mergeX `on` x) pos pos'
	, y = (mergeY `on` y) pos pos'
	}

(.+), (.-) :: Position -> Position -> Position
(.+) = mergePosition (+) (+)
(.-) = mergePosition (-) (-)

belowLeft :: Position -> Position -> Bool
belowLeft pos pos' = x pos <= x pos' && y pos <= y pos'

-- | Given a 'topRight', is the next argument inside the axis-aligned rectangle
-- from the origin to just shy of the top right?
inside :: Position -> Position -> Bool
inside extent pos = belowLeft def pos && belowLeft pos (extent .- Position 1 1)
