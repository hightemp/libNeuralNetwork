<?php

namespace libNeuralNetwork;

use Exception;

class DataFormatter
{
  public $values;
  public $indexTable;
  public $characterTable;
  public $characters;  
          
  function __construct($aValues, $iMaxThreshold = 0) 
  {
    if (!$aValues) 
      return;

    $this->values = $aValues;
    // go over all characters and keep track of all unique ones seen
    // count up all characters
    $this->indexTable = [];
    $this->characterTable = [];
    $this->characters = [];
    $this->fnBuildCharactersFromIterable($aValues);
    $this->fnBuildTables($iMaxThreshold);
  }
  
  public function fnBuildCharactersFromIterable($aValues) 
  {
    $aTempCharactersTable = [];
    for ($iDataFormatterIndex = 0, $iDataFormatterLength = count($aValues); $iDataFormatterIndex < $iDataFormatterLength; $iDataFormatterIndex++) {
      $aCharacters = $aValues[$iDataFormatterIndex];

      if (is_string($aCharacters)) {
        $aCharacters = str_split($aCharacters);
      }
      
      if (is_array($aCharacters)) {
        for ($iCharacterIndex = 0, $iCharactersLength = count($aCharacters); $iCharacterIndex < $iCharactersLength; $iCharacterIndex++) {
          $sCharacter = $aCharacters[$iCharacterIndex];
          
          if (isset($aTempCharactersTable[$sCharacter])) 
            continue;
          
          $aTempCharactersTable[$sCharacter] = true;
          array_push($this->characters, $sCharacter);
        }
      } else {
        $sCharacter = $aValues[$iDataFormatterIndex];
        
        if (isset($aTempCharactersTable[$sCharacter])) 
          continue;
        
        $aTempCharactersTable[$iDataFormatterIndex] = true;
        array_push($this->characters, $sCharacter);
      }
    }
  }

  public function fnBuildTables($iMaxThreshold) 
  {
    // filter by count threshold and create pointers
    $iCharactersLength = count($this->characters);
    for($iCharacterIndex = 0; $iCharacterIndex < $iCharactersLength; $iCharacterIndex++) {
      $sCharacter = $this->characters[$iCharacterIndex];
      if($iCharacterIndex >= $iMaxThreshold) {
        // add character to dataFormatter
        $this->indexTable[$sCharacter] = $iCharacterIndex;
        $this->characterTable[$iCharacterIndex] = $sCharacter;
      }
    }
  }

  public function fnToIndexes($mValue, $iMaxThreshold = 0) 
  {
    $aResult = [];
    $aValue = [];

    if (is_string($mValue))
      $aValue = str_split($mValue);
    
    if (is_array($mValue))
      $aValue = $mValue;
    
    for ($iI = 0, $iMax = count($aValue); $iI < $iMax; $iI++) {
      $sCharacter = $aValue[$iI];
      $iIndex = $this->indexTable[$sCharacter];
      if (!isset($this->indexTable[$sCharacter])) {
        throw new Exception(`unrecognized character "$sCharacter"`);
      }
      if ($iIndex < $iMaxThreshold) 
        continue;
      array_push($aResult, $iIndex);
    }

    return $aResult;
  }

  public function fnToIndexesInputOutput($mValue1, $mValue2 = null, $iMaxThreshold = 0)
  {
    $aResult;
    
    if (is_string($mValue1)) {
      $aResult = $this->fnToIndexes(array_merge(str_split($mValue1), ['stop-input', 'start-output']), $iMaxThreshold);
    } else {
      $aResult = $this->fnToIndexes(array_merge($mValue1, ['stop-input', 'start-output']), $iMaxThreshold);
    }
    
    if ($mValue2 === null) 
      return $aResult;

    if (is_string($mValue2)) {
      return array_merge($aResult, $this->fnToIndexes(str_split($mValue2), $iMaxThreshold));
    } else {
      return array_merge($aResult, $this->fnToIndexes($mValue2, $iMaxThreshold));
    }
  }
  
  public function fnToCharacters($aIndices, $iMaxThreshold = 0) 
  {
    $aResult = [];

    for ($iI = 0, $iMax = count($aIndices); $iI < $iMax; $iI++) {
      $iIndex = $aIndices[$iI];
      if ($iIndex < $iMaxThreshold) 
        continue;
      $sCharacter = $this->characterTable[$iIndex];
      if (!isset($this->characterTable[$iIndex])) {
        throw new Exception(`unrecognized index "$iIndex"`);
      }
      array_push($aResult, $sCharacter);
    }

    return $aResult;
  }
  
  public function fnToString($aIndices, $iMaxThreshold) 
  {
    return join('', $this->fnToCharacters($aIndices, $iMaxThreshold));
  }

  public function fnAddInputOutput() 
  {
    $this->fnAddSpecial('stop-input');
    $this->fnAddSpecial('start-output');
  }
  
  public function fnAddSpecial(...$aArguments) 
  {
    for ($iI = 0; $iI < count($aArguments); $iI++) {
      $sSpecial = $aArguments[$iI];
      $iSpecialIndex = $this->indexTable[$sSpecial] = count($this->characters);
      $this->characterTable[$iSpecialIndex] = $sSpecial;
      array_push($this->characters, $sSpecial);
    }
  }
  
  public static function fnFromAllPrintable($iMaxThreshold, $aValues = ['\n']) 
  {
    for($iI = 32; $iI <= 126; $iI++) {
      array_push($aValues, chr($iI));
    }
    return new DataFormatter($aValues, $iMaxThreshold);
  }

  public static function fnFromAllPrintableInputOutput($iMaxThreshold, $aValues = ['\n']) 
  {
    $oDataFormatter = DataFormatter::fnFromAllPrintable($iMaxThreshold, $aValues);
    $oDataFormatter->fnAddInputOutput();
    return $oDataFormatter;
  }

  public static function fnFromStringInputOutput($sString, $iMaxThreshold) 
  {
    //const values = String.prototype.concat(...new Set(string));
    $aValues = str_split($sString);
    $aValues = array_unique($aValues);
    $oDataFormatter = new DataFormatter($aValues, $iMaxThreshold);
    $oDataFormatter->fnAddInputOutput();
    return $oDataFormatter;
  }

  public static function fnFromArrayInputOutput($aArray, $iMaxThreshold) 
  {
    sort($aArray);
    $oDataFormatter = new DataFormatter($aArray, maxThreshold);
    $oDataFormatter->fnAddInputOutput();
    return $oDataFormatter;
  }

  public static function fnFromString($sString, $iMaxThreshold) 
  {
    $aValues = str_split($sString);
    $aValues = array_unique($aValues);
    return new DataFormatter($aValues, $iMaxThreshold);
  }

  public static function fnFromJSON($sJSON) 
  {
    $aJSON = json_decode($sJSON, true);
    $oDataFormatter = new DataFormatter();
    $oDataFormatter->indexTable = $aJSON['indexTable'];
    $oDataFormatter->characterTable = $aJSON['characterTable'];
    $oDataFormatter->values = $aJSON['values'];
    $oDataFormatter->characters = $aJSON['characters'];
    return $oDataFormatter;
  }  
}

