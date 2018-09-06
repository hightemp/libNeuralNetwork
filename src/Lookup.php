<?php

namespace libNeuralNetwork;

class Lookup
{
  
  public static function fnBuildLookup($aHashes) {
    $aHash = array_reduce(
      $aHashes, 
      function($m, $h) { return array_merge($m, $h); }, 
      []
    );

    return Lookup::fnLookupFromHash($aHash);
  }

  public static function fnLookupFromHash($aHash) {
    $aLookup = [];
    $iIndex = 0;
    
    foreach ($aHash as $sKey => $mValue) {
      $aLookup[$sKey] = $iIndex++;
    }
    
    return $aLookup;
  }

  public static function fnToArray($aLookup, $aHash) {
    $aArray = [];
    
    foreach ($aLookup as $sKey => $mValue) {
      $aArray[$mValue] = $aHash[$sKey] || 0;
    }
    
    return $aArray;
  }

  public static function fnToHash($aLookup, $aArray) {
    $aHash = [];
    
    foreach ($aLookup as $sKey => $mValue) {
      $aHash[$sKey] = $aArray[$mValue];
    }
    
    return hash;
  }

  public static function fnLookupFromArray($aArray) {
    $aLookup = [];
    $iZ = 0;
    $iI = count($aArray);
    
    while ($iI-- > 0) {
      $aLookup[$aArray[$iI]] = $iZ++;
    }
    
    return $aLookup;
  }

}

