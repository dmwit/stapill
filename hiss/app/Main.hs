{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Control.Applicative (liftA2)
import Control.Monad (replicateM_)

import Crypto.Hash.SHA256 (hash)

import Data.Aeson
import Data.Aeson.TH (deriveJSON)
import Data.Bits (shiftL, shiftR, (.|.), (.&.))
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Functor ((<&>))
import Data.Int (Int64)
import Data.List (findIndex, nub, (\\))
import Data.Maybe (fromJust)
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import qualified Data.Map as M
import Data.Text (Text)
import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.Word (Word32)

import Dr.Mario.Model
import Dr.Mario.Pathfinding
import JSONTH

import GHC.Stack (HasCallStack)

import System.IO (BufferMode(LineBuffering), hSetBuffering, stdout)
import System.Random.MWC (GenIO, Uniform, uniformM, uniformRM)
import qualified System.Random.MWC as MWC

import Text.Printf (printf)

data Request
  = RequestReset { resetSeed :: !Int64, resetLevel :: !Int }
  | RequestStep { stepAction :: !Int }
  | RequestClose
  deriving (Eq, Ord, Read, Show)

deriveJSON strictPrefixOptions ''Request

data ImmutableState = ImmutableState
  { immViruses :: Int
  , immPillsUsed :: Int
  , immFrame :: Int
  , immCoarseSpeed :: CoarseSpeed
  , immLookahead :: Lookahead
  , immPaths :: HashMap MidPlacement MidPath
  } deriving (Eq, Ord, Read, Show)

data SimState = SimState
  { simBoard :: IOBoard
  , simGameRNG :: GenIO
  , simPlayerRNG :: GenIO
  , simOther :: IORef ImmutableState
  }

-- | Represents a 3D tensor of size indexCount x length indices x length (head
-- indices), where result[i][j][k] = i == indices[j][k].
data OneHot2D = OneHot2D
  { oneHot2DIndexCount :: Int
  , oneHot2DIndices :: V.Vector (V.Vector Int)
  } deriving (Eq, Ord, Read, Show)

deriveJSON strictPrefixOptions { fieldLabelModifier = stripNWords 4 } ''OneHot2D

data Observation = Observation
  { observationColor :: OneHot2D
  , observationShape :: OneHot2D
  , observationOther :: HashMap Text Double
  } deriving (Eq, Ord, Read, Show)

deriveJSON strictPrefixOptions ''Observation

data Reply
  = ReplyError { errorDescription :: String }
  | ReplySuccess
    { successObservation :: Observation
    , successReward :: HashMap Text Double
    , successValidActions :: V.Vector Int
    , successTerminated :: Bool
    , successColors :: [Color]
    }
  deriving (Eq, Ord, Read, Show)

deriveJSON strictPrefixOptions ''Reply

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  linewise uninit
  where
  reply = BL8.putStrLn . encode
  complain err = reply ReplyError { errorDescription = err }
  linewise act = do
    line <- BS8.getLine
    case eitherDecode (BL8.fromStrict line) of
      Left err -> complain err >> linewise act
      Right req -> act req

  uninit = \case
    RequestClose {} -> pure ()
    RequestStep {} -> do
      complain "step requested before reset"
      linewise uninit
    RequestReset { resetSeed = seed, resetLevel = level } -> do
      sim <- newSimState seed level
      succeed sim HM.empty False

  running sim = \case
    RequestClose {} -> pure ()
    req@RequestReset {} -> uninit req
    RequestStep { stepAction = action } -> do
      (requestedPill, actualPill@pill, path) <- validateAction sim mp_
      rAction <- actionReward sim requestedPill actualPill
      -- TODO: look for HasCallStack and turn it into an error reply instead of a crash
      result <- stepSimState sim pill path
      succeed sim
        (rAction <> outcomeReward result)
        (resultWon result || resultLost result)
      where
      mp_ = parseAction action

  succeed sim reward terminated = do
    colors <- chooseColorOrder sim
    observation <- observe sim colors
    actions <- actionIndices sim
    reply ReplySuccess
      { successObservation = observation
      , successReward = reward
      , successValidActions = actions
      , successTerminated = terminated
      , successColors = colors
      }
    linewise (running sim)

newSimState :: Int64 -> Int -> IO SimState
newSimState seed level = do
  gameRNG <- splitSeed seed (BS8.pack "level")
  playerRNG <- splitSeed seed (BS8.pack "moves")
  nesRNG <- uniformRM (2, maxBound) gameRNG >>= mnewRNG
  board <- munsafeRandomBoard nesRNG level
  lk <- uniformM gameRNG
  firstFrame <- uniformRM (0, 1) gameRNG
  coarseSpeed <- ([Med, Hi]!!) <$> uniformRM (0, 1) gameRNG

  other <- newIORef ImmutableState
    { immViruses = max 4 . min 84 $ level * 4
    , immPillsUsed = 0
    , immFrame = firstFrame
    , immCoarseSpeed = coarseSpeed
    , immPaths = error "immPaths accessed before pathfinding was executed (newSimState)"
    , immLookahead = lk
    }
  let sim = SimState
        { simBoard = board
        , simGameRNG = gameRNG
        , simPlayerRNG = playerRNG
        , simOther = other
        }
  sim <$ simReachable sim

data SimStepResults = SimStepResults
  { resultCleanup :: Maybe CleanupResults -- ^ redundant with 'resultClears', but might be handy
  , resultClears :: Maybe ClearResults
  , resultFrames :: Int
  , resultWon :: Bool
  , resultLost :: Bool
  } deriving (Eq, Ord, Read, Show)

stepSimState :: SimState -> Pill -> MidPath -> IO SimStepResults
stepSimState sim pill path = do
  imm <- readIORef (simOther sim)
  details <- mplaceDetails (simBoard sim) pill
  lk <- uniformM (simGameRNG sim)

  let cleanup = summarizeClearResults <$> details
      dframes = mpPathLength path + interControlFrameCost
        (approxFrameCostModel (immCoarseSpeed imm))
        (y (bottomLeftPosition pill))
        cleanup
      imm' = ImmutableState
        { immViruses = immViruses imm - maybe 0 clears cleanup
        , immPillsUsed = immPillsUsed imm + 1
        , immFrame = immFrame imm + dframes
        , immCoarseSpeed = immCoarseSpeed imm
        , immLookahead = lk
        , immPaths = error "immPaths accessed before pathfinding was executed (stepSimState)"
        }
  writeIORef (simOther sim) imm'
  paths <- simReachable sim

  return SimStepResults
    { resultCleanup = cleanup
    , resultClears = details
    , resultFrames = dframes
    , resultWon = immViruses imm' == 0
    , resultLost = HM.size paths == 0
    }

-- | Updates the reachable placements, and returns them too.
simReachable :: SimState -> IO (HashMap MidPlacement MidPath)
simReachable sim = do
  imm <- readIORef (simOther sim)
  paths <- mapproxReachable (simBoard sim) (immFrame imm .&. 1 == 0) (gravity (immCoarseSpeed imm) (immPillsUsed imm))
  paths <$ writeIORef (simOther sim) imm { immPaths = paths }

validateAction :: HasCallStack => SimState -> MidPlacement -> IO (Pill, Pill, MidPath)
validateAction sim mp = do
  imm <- readIORef (simOther sim)
  let paths = immPaths imm
      pill mp' = mpPill mp' (immLookahead imm)
      pillify (mp', path) = (pill mp, pill mp', path)
  pillify <$> case HM.lookup mp paths of
    Just path -> pure (mp, path)
    Nothing -> (HM.toList paths !!) <$> uniformRM (0, HM.size paths - 1) (simPlayerRNG sim)

actionReward :: SimState -> Pill -> Pill -> IO (HashMap Text Double)
actionReward sim reqPill pill = do
  horzReward <- sumFor placements \(pos, cell) ->
    sumFor [0..7] \x' -> do
      nearbyCell <- mget (simBoard sim) pos { x = x' }
      pure $ if (nearbyCell >>= color) == color cell then 0.5 ^^ abs (x' - x pos) else 0
  vertReward <- sumFor placements \(pos, cell) ->
    sumFor [0..15] \y' -> do
      nearbyCell <- mget (simBoard sim) pos { y = y' }
      pure $ if (nearbyCell >>= color) == color cell then 0.5 ^^ abs (y' - y pos) else 0
  chasm <- case orientation (content pill) of
    Vertical -> pure 0
    Horizontal -> sumFor placements \(pos, _) -> countEmpties (simBoard sim) (unsafeMove down pos)
  pure $ HM.fromList
    [ ("match/horz", horzReward)
    , ("match/vert", vertReward)
    , ("invalid", true1 (reqPill /= pill))
    , ("chasm", fromIntegral chasm)
    ]
  where
  placements =
    [ (bottomLeftPosition pill, bottomLeftCell (content pill))
    , (otherPosition pill, otherCell (content pill))
    ]
  sumFor xs f = sum <$> traverse f xs

outcomeReward :: SimStepResults -> HashMap Text Double
outcomeReward result = HM.fromList
  [ ("frames", fromIntegral (resultFrames result))
  , ("viruses", sum (fromIntegral . clears <$> resultCleanup result))
  , ("halves", sum (fromIntegral . countHalves <$> resultClears result))
  , ("win", true1 (resultWon result))
  , ("lose", true1 (resultLost result))
  ]

countHalves :: ClearResults -> Int
countHalves = goClear where
  goClear = \case
    NoClear -> 0
    Clear ocs ds -> M.size (M.filter (\oc -> oshape oc /= Virus) ocs) + goDrops ds

  goDrops = \case
    NoDrop -> 0
    Drop _ cs -> goClear cs

countEmpties :: IOBoard -> Position -> IO Int
countEmpties board pos = mget board pos >>= \case
  Just Empty -> succ <$> countEmpties board (unsafeMove down pos)
  _ -> pure 0

true1 :: Bool -> Double
true1 b = if b then 1 else 0

splitSeed :: Int64 -> BS.ByteString -> IO GenIO
splitSeed n tag = do
  gen <- MWC.initialize . bsToWord32s . hash $ int64ToByteString n <> tag
  gen <$ replicateM_ 128 (uniformM @Int gen)

int64ToByteString :: Int64 -> BS.ByteString
int64ToByteString = BS.pack . map fromIntegral . take 8 . iterate (`shiftR` 8)

bsToWord32s :: BS.ByteString -> Vector Word32
bsToWord32s bs = V.generate ((BS.length bs + 3) `shiftR` 2) \i_ -> let i = i_ `shiftL` 2 in
  BS.foldl' (\a b -> shiftL a 8 .|. fromIntegral b) 0 . BS.take 4 . BS.drop i $ bs

actionIndices :: SimState -> IO (Vector Int)
actionIndices sim = do
  paths <- immPaths <$> readIORef (simOther sim)
  pure . V.fromListN (HM.size paths) . map ppAction $ HM.keys paths

ppAction :: MidPlacement -> Int
ppAction mp = ppPosition (mpBottomLeft mp) .|. shiftL (mpRotations mp) 7

parseAction :: Int -> MidPlacement
parseAction n = MidPlacement
  { mpBottomLeft = parsePosition n
  , mpRotations = (n `shiftR` 7) .&. 0x3
  }

observe :: SimState -> [Color] -> IO Observation
observe sim colors = do
  oShape <- observeShape (simBoard sim)
  oColor <- observeColor (simBoard sim) colors
  oDouble <- observeDouble sim
  oGravity <- observeGravity (simOther sim)
  pure Observation
    { observationShape = oShape
    , observationColor = oColor
    , observationOther = oDouble <> oGravity
    }

-- Build the color permutation: left pill half is 1, right pill half is 2, last
-- color is 3. Doubles rerandomize the right half.
chooseColorOrder :: SimState -> IO [Color]
chooseColorOrder sim = do
  lk <- simLookahead sim
  let pillColors = nub [leftColor lk, rightColor lk]
      otherColors = [minBound..maxBound] \\ pillColors
  swap <- uniformM (simGameRNG sim)
  pure $ pillColors ++ (if swap then reverse else id) otherColors

observeColor :: HasCallStack => IOBoard -> [Color] -> IO OneHot2D
observeColor b colors = observeBoard b indexCountColor (indexColor . ocolor) where
  indexColor = \case Blue -> blue; Red -> red; Yellow -> yellow
  [blue, red, yellow] = [fromJust (findIndex (c==) colors) | c <- [Blue, Red, Yellow]]

observeShape :: HasCallStack => IOBoard -> IO OneHot2D
observeShape b = observeBoard b indexCountShape (indexShape . oshape)

-- | The given function should return nonnegative 'Int's in the range [0,
-- arg2). The Observation produced will be a 'OneHot2D' with 0 for empties.
observeBoard :: HasCallStack => IOBoard -> Int -> (OCell -> Int) -> IO OneHot2D
observeBoard b count f = do
  values <- V.generateM nesWidth \bx ->
    V.generateM nesHeight \by ->
      mget b (Position bx by) >>= \case
        Nothing -> fail . printf "board was too small, expected %dx%d, saw\n%s" nesWidth nesHeight =<< mppST b
        Just Empty -> pure 0
        Just (Occupied c s) -> pure . succ $ f (OCell c s)
  pure OneHot2D
    { oneHot2DIndexCount = succ count
    , oneHot2DIndices = values
    }
  where
  nesWidth = 8
  nesHeight = 16

observeDouble :: SimState -> IO (HashMap Text Double)
observeDouble sim = simLookahead sim <&> \lk ->
  HM.singleton "double" . fromIntegral . fromEnum $ leftColor lk == rightColor lk

observeGravity :: IORef ImmutableState -> IO (HashMap Text Double)
observeGravity ref = readIORef ref <&> \imm -> ppGravity (gravity (immCoarseSpeed imm) (immPillsUsed imm))

ppGravity :: Int -> (HashMap Text Double)
ppGravity n = HM.fromList
  [ ("superframe/row", fromIntegral n / 40)
  , ("row/frame", recip (fromIntegral n))
  ]

simLookahead :: SimState -> IO Lookahead
simLookahead sim = immLookahead <$> readIORef (simOther sim)

indexCountColor :: Int
indexCountColor = 3

indexCountShape :: Int
indexCountShape = 4

indexShape :: Shape -> Int
indexShape = \case
  Virus -> 0
  Disconnected -> 1
  North -> 1
  South -> 1
  East -> 2
  West -> 3

ppPosition :: Position -> Int
ppPosition pos = y pos + shiftL (x pos) 4

parsePosition :: Int -> Position
parsePosition n = Position
  { x = (n `shiftR` 4) .&. 0x7
  , y = (n `shiftR` 0) .&. 0xf
  }

instance Uniform Lookahead where uniformM g = liftA2 Lookahead (uniformM g) (uniformM g)
instance Uniform Color where uniformM g = toEnum <$> uniformRM (0, 2) g
