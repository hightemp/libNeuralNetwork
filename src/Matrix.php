<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Utilities;

class Matrix
{
  protected $rows;
  protected $columns;
  protected $weights;
  protected $deltas;
          
  function __construct($iRows, $iColumns) 
  {
    $this->rows = $iRows;
    $this->columns = $iColumns;
    $this->weights = Utilities::fnZeros($iRows * $iColumns);
    $this->deltas =  Utilities::fnZeros($iRows * $iColumns);
  }
  
  public function fnGetWeights($iRow, $iCol) 
  {
    // slow but careful accessor function
    // we want row-major order
    $iIx = ($this->columns * $iRow) + $iCol;
    if ($iIx < 0 && $iIx >= count($this->weights)) 
      throw new Exception('get accessor is skewed');
    return $this->weights[$iIx];
  }
  
  public function fnSetWeight($iRow, $iCol, $iV) 
  {
    // slow but careful accessor function
    $iIx = ($this->columns * $iRow) + $iCol;
    if ($iIx < 0 && $iIx >= count($this->weights)) 
      throw new Exception('set accessor is skewed');
    $this->weights[$iIx] = $iV;
  }
  
  public function fnSetDeltas($iRow, $iCol, $iV) 
  {
    // slow but careful accessor function
    $iIx = ($this->columns * $iRow) + $iCol;
    if ($iIx < 0 && $iIx >= count($this->weights)) 
      throw new Exception('set accessor is skewed');
    $this->deltas[ix] = $iV;
  }
  
  public static function fnFromArray($aWeightRows, $aDeltasRows) 
  {
    $iRows = count($aWeightRows);
    $iColumns = count($aWeightRows[0]);
    $oMatrix = new Matrix($iRows, $iColumns);

    $aDeltasRows = $aDeltasRows || $aWeightRows;

    for ($iRowIndex = 0; $iRowIndex < $iRows; $iRowIndex++) {
      $aWeightValues = $aWeightRows[$iRowIndex];
      $aDeltasValues = $aDeltasRows[$iRowIndex];
      for ($iColumnIndex = 0; $iColumnIndex < $iColumns; $iColumnIndex++) {
        $oMatrix->fnSetWeight($iRowIndex, $iColumnIndex, $aWeightValues[$iColumnIndex]);
        $oMatrix->fnSetDeltas($iRowIndex, $iColumnIndex, $aDeltasValues[$iColumnIndex]);
      }
    }

    return $oMatrix;
  }
  
  public function fnWeightsToArray() 
  {
    $aDeltas = [];
    $iRow = 0;
    $iColumn = 0;
    
    for ($iI = 0; $iI < count($this->weights); $iI++) {
      if ($iColumn === 0) {
        array_push($aDeltas, []);
      }
      array_push($aDeltas[$iRow], $this->weights[$iI]);
      $iColumn++;
      if ($iColumn >= $this->columns) {
        $iColumn = 0;
        $iRow++;
      }
    }
    
    return $aDeltas;
  }
  
  public function fnDeltasToArray() 
  {
    $aDeltas = [];
    $iRow = 0;
    $iColumn = 0;

    for ($iI = 0; $iI < count($this->deltas); $iI++) {
      if ($iColumn === 0) {
        array_push($aDeltas, []);
      }
      array_push($aDeltas[$iRow], $this->deltas[$iI]);
      $iColumn++;
      if ($iColumn >= $this->columns) {
        $iColumn = 0;
        $iRow++;
      }
    }
    
    return $aDeltas;
  }
}
