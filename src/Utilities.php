<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Lookup;

class Utilities
{
  public static function fnMax($aValues) 
  {
    return max(self::fnToArray($aValues));
  }
  
  public static function fnMse($aErrors) 
  {
    // mean squared error
    $iSum = 0;
    for ($iI = 0; $iI < count($aErrors); $iI++) {
      $iSum += pow($aErrors[$iI], 2);
    }
    return $iSum / count($aErrors);
  }
  
  public static function fnOnes($iSize)
  {
    return array_fill(0, $iSize, 1);
  }
  
  public static function fnZeros($iSize)
  {
    return array_fill(0, $iSize, 0);
  }
  
  public static function fnRandomWeight() 
  {
    return rand(0, 100000) / 100000 * 0.4 - 0.2;
  }
  
  public static function fnRandos($iSize) 
  {
    $aArray = array_fill(0, $iSize, 0);
    for ($iI = 0; $iI < $iSize; $iI++) {
      $aArray[$iI] = self::fnRandomWeight();
    }
    return $aArray;
  }
  
  public static function fnToArray($mValues) 
  {
    $aKeys = array_keys($mValues);
    $aResult = [];
    
    foreach ($aKeys as $iIndex => $sKey) {
      $aResult[$iIndex] = $mValues[$sKey];
    }
    
    return $aResult;
  }
  
  public static function likely($aInput, $oNet) 
  {
    $aOutput = $oNet->run($aInput);
    $iMaxProp = null;
    $iMaxValue = -1;
    foreach ($aOutput as $iProp) {
      $iValue = $aOutput[$iProp];
      if ($iValue > $iMaxValue) {
        $iMaxProp = $iProp;
        $iMaxValue = $iValue;
      }
    }
    return $iMaxProp;
  }
}
