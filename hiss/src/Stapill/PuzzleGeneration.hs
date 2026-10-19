{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE BlockArguments #-}

module Stapill.PuzzleGeneration (
	MonadDist (..),
	ensure,
	GenDist (..),
	runGenDist,
	sampleGenDist,
	RejectionFraction (..),
	) where

import Control.Applicative (empty, liftA2)
import Control.Monad (join, unless)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Maybe (MaybeT (..))
import Control.Monad.Trans.Reader (ReaderT (..), ask)
import Data.Foldable (toList)
import Data.Word (Word32)
import System.Random.MWC (GenIO)
import qualified System.Random.MWC as MWC

-- Writer (Product Rational) a, but with better names for the fields
data Category a = Category
	{ weight :: Rational
	, label :: a
	} deriving (Eq, Ord, Read, Show, Foldable, Functor, Traversable)

instance Applicative Category where
	pure = Category 1
	Category wf f <*> Category wa a = Category (wf * wa) (f a)

instance Monad Category where
	Category wouter outer >>= f = inner { weight = weight inner * wouter } where inner = f outer

-- | A monad with choices and an explicitly representable failed branch.
class Monad m => MonadDist m where
	-- | Choose an element, with non-negative weights. This is checked. The
	-- weights need not sum to one.
	categorical :: Foldable f => f (Category a) -> m a
	categorical choices = case normalizeWeights choices of
		[] -> reject
		probs -> unsafeCategorical probs

	-- | Choose an element. The weights must be positive and must sum to oneG
	-- This is not checked.
	unsafeCategorical :: [Category a] -> m a

	-- | Choose uniformly from the supplied alternatives.
	select :: (Foldable f, Functor f) => f a -> m a
	select = categorical . fmap (Category 1)

	-- | Produce no result.
	reject :: m a

ensure :: MonadDist m => (a -> Bool) -> a -> m a
ensure p x = x <$ unless (p x) reject

instance MonadDist [] where
	-- Lists enumerate possibilities; they do not model weighted sampling.
	unsafeCategorical = fmap label
	reject = []

-- | A random generator with rejection as failure.
newtype GenDist a = GenDist
	{ unGenDist :: ReaderT GenIO (MaybeT IO) a
	} deriving (Functor, Applicative, Monad)

-- | Run a generator.  A 'Nothing' result means that generation rejected.
runGenDist :: GenIO -> GenDist a -> IO (Maybe a)
runGenDist gen (GenDist action) = runMaybeT (runReaderT action gen)

-- | Run a generator repeatedly until it does not reject.
sampleGenDist :: GenIO -> GenDist a -> IO a
sampleGenDist gen act = go where go = runGenDist gen act >>= maybe go pure

instance MonadDist GenDist where
	reject = GenDist empty
	unsafeCategorical probs = GenDist do
		u <- liftIO . MWC.uniformM @Word32 =<< ask
		let target = toRational u / (toRational (maxBound @Word32))
		pure (chooseAt target probs)

-- | An exact distribution whose missing mass is rejection.
--
-- The two fields are kept together so the representation can be inspected in
-- tests and by callers that want to reason about rejection.  The constructor
-- is not smart, but values produced by 'MonadDist' have successful and
-- rejected mass summing to one.
data RejectionFraction a = RejectionFraction
	{ samples :: [Category a]
	, rejection :: Rational
	} deriving (Eq, Ord, Read, Show, Functor)

instance Applicative RejectionFraction where
	pure a = RejectionFraction (pure (pure a)) 0
	f <*> x = RejectionFraction
		{ samples = liftA2 (<*>) (samples f) (samples x)
		, rejection = rejection f + (1 - rejection f) * rejection x
		}

instance Monad RejectionFraction where rf >>= k = joinRF (k <$> rf)
instance MonadDist RejectionFraction where
	reject = RejectionFraction [] 1
	unsafeCategorical = flip RejectionFraction 0

joinRF :: RejectionFraction (RejectionFraction a) -> RejectionFraction a
joinRF rf = RejectionFraction
	{ samples = map join . traverse samples =<< samples rf
	, rejection = rejection rf + sum [p * rejection xs | Category p xs <- samples rf]
	}

normalizeWeights :: Foldable f => f (Category a) -> [Category a]
normalizeWeights choices
	| any ((< 0) . weight) choices = error "categorical: negative weight"
	| otherwise = let ws = filter ((> 0) . weight) (toList choices)
		in [Category (w / sum (map weight ws)) a | Category w a <- ws]

chooseAt :: Rational -> [Category a] -> a
chooseAt remaining = \case
	[] -> error "chooseAt: target weight exceeds total weight"
	cat:rest
		| remaining <= weight cat -> label cat
		| otherwise -> chooseAt (remaining - weight cat) rest
