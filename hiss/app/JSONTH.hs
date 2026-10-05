module JSONTH (stripFirstWord, stripNWords, strictPrefixOptions) where

import Data.Aeson
import Data.Char (isLower, toLower)
import GHC.Stack (HasCallStack, callStack, prettyCallStack)
import Text.Printf

stripFirstWord :: HasCallStack => String -> String
stripFirstWord s = case span isLower (drop 1 s) of
  (_, c:cs) -> toLower c : cs
  -- I'd like to call real error here, but deriveJSON seems to force this
  -- result even if it doesn't use it. When the type name is a single word
  -- and is not a sum, that's a problem. So as a second best solution, we'll
  -- just try to make it as obvious as possible to anybody that inspects the
  -- resulting JSON.
  (_, "") -> printf "ERROR: stripFirstWord used on name \"%s\", which has no second word\n%s"
    s (prettyCallStack callStack)

stripNWords :: HasCallStack => Int -> String -> String
stripNWords n s = iterate stripFirstWord s !! n

strictPrefixOptions :: Options
strictPrefixOptions = defaultOptions
  { fieldLabelModifier = stripFirstWord
  , constructorTagModifier = stripFirstWord
  , allNullaryToStringTag = False
  , allowOmittedFields = False
  , sumEncoding = ObjectWithSingleField
  , rejectUnknownFields = True
  }
