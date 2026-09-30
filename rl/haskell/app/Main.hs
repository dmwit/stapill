{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Control.Exception (SomeException, displayException, try)
import Data.Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.List (find)
import Data.Maybe (mapMaybe)
import qualified Data.HashMap.Strict as HM
import qualified Data.Map.Strict as M
import qualified Data.Vector as V
import Data.Word (Word16)
import Dr.Mario.Model
import Dr.Mario.Pathfinding
import GHC.Generics (Generic)
import System.IO (BufferMode(LineBuffering), hSetBuffering, stdout)

data Request
  = ResetRequest { resetSeed :: !Word16, resetLevel :: !Int }
  | StepRequest { stepAction :: !Int }
  | CloseRequest
  deriving (Eq, Show, Generic)

instance FromJSON Request where
  parseJSON = withObject "Request" $ \o -> do
    kind <- o .: "type"
    case (kind :: String) of
      "reset" -> ResetRequest <$> o .:? "seed" .!= 1 <*> o .:? "level" .!= 0
      "step" -> StepRequest <$> o .: "action"
      "close" -> pure CloseRequest
      _ -> fail "unknown request type"

data SimState = SimState
  { simLevel :: !Int
  , simLookaheads :: !(V.Vector Lookahead)
  , simLookaheadIndex :: !Int
  , simBoard :: !Board
  , simLockedPills :: !Int
  , simFrames :: !Int
  , simFallbackSeed :: !Word16
  , simDone :: !Bool
  , simWon :: !Bool
  }

data Action = Action
  { actionIndex :: !Int
  , actionPill :: !Pill
  , actionFrames :: !Int
  }

data Reply = Reply
  { replyType :: !String
  , replyObservation :: ![Int]
  , replyMask :: ![Bool]
  , replyReward :: !Double
  , replyTerminated :: !Bool
  , replyTruncated :: !Bool
  , replyInfo :: !Value
  }

instance ToJSON Reply where
  toJSON reply = object
    [ "type" .= replyType reply
    , "observation" .= replyObservation reply
    , "action_mask" .= replyMask reply
    , "reward" .= replyReward reply
    , "terminated" .= replyTerminated reply
    , "truncated" .= replyTruncated reply
    , "info" .= replyInfo reply
    ]

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  loop Nothing
  where
    loop :: Maybe SimState -> IO ()
    loop state = do
      line <- BS.getLine
      if BS.null line
        then pure ()
        else do
          result <- try (handle state (eitherDecode (BL.fromStrict line)))
            :: IO (Either SomeException (Maybe SimState, Maybe Reply))
          case result of
            Left err -> do
              BL.putStrLn (encode (errorReply (displayException err)))
              loop state
            Right (nextState, reply) -> do
              maybe (pure ()) (BL.putStrLn . encode) reply
              case reply of
                Nothing -> pure ()
                Just _ -> loop nextState

handle :: Maybe SimState -> Either String Request -> IO (Maybe SimState, Maybe Reply)
handle _ (Left err) = pure (Nothing, Just (errorReply err))
handle _ (Right (ResetRequest seed level)) = do
  let (lookaheads, board) = randomLevel seed level
      state = SimState level lookaheads 0 board 0 0 seed False False
  pure (Just state, Just (stateReply "reset" state 0 False False []))
handle Nothing (Right (StepRequest _)) =
  pure (Nothing, Just (errorReply "step requested before reset"))
handle (Just state) (Right (StepRequest action)) = do
  let (state', reply) = stepState state action
  pure (Just state', Just reply)
handle _ (Right CloseRequest) = pure (Nothing, Nothing)

errorReply :: String -> Reply
errorReply message = Reply
  { replyType = "error"
  , replyObservation = []
  , replyMask = replicate actionCount False
  , replyReward = 0
  , replyTerminated = False
  , replyTruncated = False
  , replyInfo = object ["error" .= message]
  }

actionCount :: Int
actionCount = 8 * 16 * 4

stateReply :: String -> SimState -> Double -> Bool -> Bool -> [(String, Value)] -> Reply
stateReply kind state reward terminated truncated extra = Reply
  { replyType = kind
  , replyObservation = observation state
  , replyMask = legalMask state
  , replyReward = reward
  , replyTerminated = terminated
  , replyTruncated = truncated
  , replyInfo = object
      ( [ "level" .= simLevel state
        , "viruses" .= countViruses (simBoard state)
        , "frames" .= simFrames state
        , "locked_pills" .= simLockedPills state
        ] ++ map (\(key, value) -> Key.fromString key .= value) extra
      )
  }

stepState :: SimState -> Int -> (SimState, Reply)
stepState state requested =
  let beforeBoard = simBoard state
      actions = legalActions state
      (chosen, replaced, fallbackSeed') = chooseAction state requested actions
  in case chosen of
    Nothing ->
      let state' = state { simDone = True, simWon = False, simFallbackSeed = fallbackSeed' }
          info = [("action_was_invalid", Bool True), ("loss_reason", String "no_legal_actions")]
      in (state', stateReply "step" state' (-100) True False info)
    Just action ->
      case placeDetails beforeBoard (actionPill action) of
        Nothing ->
          let state' = state { simDone = True, simWon = False, simFallbackSeed = fallbackSeed' }
              info = [("action_was_invalid", Bool True), ("loss_reason", String "placement_rejected")]
          in (state', stateReply "step" state' (-100) True False info)
        Just (clearResult, afterBoard) ->
          let clearedNonViruses = countClearedNonViruses clearResult
              clearedViruses = countViruses beforeBoard - countViruses afterBoard
              alignmentDelta = alignmentScore afterBoard - alignmentScore beforeBoard
              nextIndex = simLookaheadIndex state
              state' = state
                { simBoard = afterBoard
                , simLookaheadIndex = nextIndex + 1
                , simLockedPills = simLockedPills state + 1
                , simFrames = simFrames state + actionFrames action
                , simFallbackSeed = fallbackSeed'
                }
              won = countViruses afterBoard == 0
              state'' = state' { simDone = won, simWon = won }
              reward = 0.1
                     + 0.25 * fromIntegral clearedNonViruses
                     + 0.05 * fromIntegral alignmentDelta
                     + 2.0 * fromIntegral clearedViruses
                     + if won then 100 else 0
                     - 0.001 * fromIntegral (actionFrames action)
              info =
                [ ("action_was_invalid", Bool replaced)
                , ("requested_action", toJSON requested)
                , ("executed_action", toJSON (actionIndex action))
                , ("frames_elapsed", toJSON (actionFrames action))
                , ("cleared_non_viruses", toJSON clearedNonViruses)
                , ("cleared_viruses", toJSON clearedViruses)
                , ("alignment_delta", toJSON alignmentDelta)
                , ("won", Bool won)
                ]
          in (state'', stateReply "step" state'' reward won False info)

chooseAction :: SimState -> Int -> [Action] -> (Maybe Action, Bool, Word16)
chooseAction state requested actions =
  case find ((== requested) . actionIndex) actions of
    Just action -> (Just action, False, simFallbackSeed state)
    Nothing
      | null actions -> (Nothing, True, simFallbackSeed state)
      | otherwise ->
          let nextSeed = advanceRNG (simFallbackSeed state)
              index = fromIntegral nextSeed `mod` length actions
          in (Just (actions !! index), True, nextSeed)

legalMask :: SimState -> [Bool]
legalMask state =
  let legal = legalActions state
      indices = map actionIndex legal
  in [index `elem` indices | index <- [0 .. actionCount - 1]]

legalActions :: SimState -> [Action]
legalActions state = mapMaybe (toAction lookahead) (HM.toList reachable)
  where
    lookahead = currentLookahead state
    reachable = unsafeApproxReachable (simBoard state) (launchPill lookahead)

    toAction lk (candidate, move) = do
      index <- encodeAction lk (content candidate) (bottomLeftPosition candidate)
      pure Action
        { actionIndex = index
        , actionPill = candidate
        , actionFrames = 1 + abs (xDelta move) + abs (yDelta move)
                       + maybe 0 (const 1) (finalRotation move)
        }

currentLookahead :: SimState -> Lookahead
currentLookahead state =
  simLookaheads state V.! (simLookaheadIndex state `mod` V.length (simLookaheads state))

encodeAction :: Lookahead -> PillContent -> Position -> Maybe Int
encodeAction lookahead content position = do
  if x position >= 0 && x position < 8 && y position >= 0 && y position < 16
    then pure ()
    else Nothing
  config <- case (orientation content, bottomLeftColor content, otherColor content) of
    (Horizontal, l, r) | (l, r) == (leftColor lookahead, rightColor lookahead) -> Just 0
                        | (l, r) == (rightColor lookahead, leftColor lookahead) -> Just 1
    (Vertical, l, r) | (l, r) == (leftColor lookahead, rightColor lookahead) -> Just 2
                      | (l, r) == (rightColor lookahead, leftColor lookahead) -> Just 3
    _ -> Nothing
  pure (config * 128 + y position * 8 + x position)

observation :: SimState -> [Int]
observation state = boardValues (simBoard state)
  ++ [colorCode (leftColor lookahead), colorCode (rightColor lookahead)]
  ++ [simLevel state, countViruses (simBoard state), simLockedPills state]
  where
    lookahead = currentLookahead state

boardValues :: Board -> [Int]
boardValues board =
  [ cellCode (unsafeGet board (Position x y))
  | y <- [15,14..0], x <- [0..7]
  ]

cellCode :: Cell -> Int
cellCode Empty = 0
cellCode (Occupied color shape) = 1 + fromEnum color * 6 + fromEnum shape

colorCode :: Color -> Int
colorCode color = 1 + fromEnum color

countClearedNonViruses :: ClearResults -> Int
countClearedNonViruses NoClear = 0
countClearedNonViruses (Clear cells drops) =
  length [() | OCell _ shape <- M.elems cells, shape /= Virus]
  + countDropped drops
  where
    countDropped NoDrop = 0
    countDropped (Drop _ clears) = countClearedNonViruses clears

alignmentScore :: Board -> Int
alignmentScore board =
  sum [sameColor (Position x y) (Position (x + 1) y) | x <- [0..6], y <- [0..15]]
  + sum [sameColor (Position x y) (Position x (y + 1)) | x <- [0..7], y <- [0..14]]
  where
    sameColor p q = case (getColor board p, getColor board q) of
      (Just a, Just b) | a == b -> 1
      _ -> 0
