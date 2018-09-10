<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Utilities;
use Exception;

class Matrix
{
  public $rows;
  public $columns;
  public $weights;
  public $deltas;
          
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
  
  public static function fnAdd(&$oProduct, $oLeft, $oRight) 
  {
    for($iI = 0; $iI < count($oLeft->weights); $iI++) {
      $oProduct->weights[$iI] = $oLeft->weights[$iI] + $oRight->weights[$iI];
      $oProduct->deltas[$iI] = 0;
    }
  }
  
  public static function fnAddB($oProduct, &$oLeft, &$oRight) 
  {
    for($iI = 0; $iI < count($oProduct->deltas); $iI++) {
      $oLeft->deltas[$iI] = $oProduct->deltas[$iI];
      $oRight->deltas[$iI] = $oProduct->deltas[$iI];
    }
  }
  
  public static function fnAllOnes(&$oProduct) 
  {
    for($iI = 0; $iI < count($oProduct->weights); $iI++) {
      $oProduct->weights[$iI] = 1;
      $oProduct->deltas[$iI] = 0;
    }
  }
  
  public static function fnCloneNegative(&$oProduct, $oLeft) 
  {
    $oProduct->rows = $oLeft->rows;
    $oProduct->columns = $oLeft->columns;
    $oProduct->weights = $oLeft->weights;
    $oProduct->deltas = $oLeft->deltas;
    for($iI = 0; $iI < count($oLeft->weights); $iI++) {
      $oProduct->weights[$iI] = -$oLeft->weights[$iI];
      $oProduct->deltas[$iI] = 0;
    }
  }
  
  public static function fnMultiply(&$oProduct, &$oLeft, &$oRight) 
  {
    $iLeftRows = $oLeft->rows;
    $iLeftColumns = $oLeft->columns;
    $iRightColumns = $oRight->columns;

    // loop over rows of left
    for($iLeftRow = 0; $iLeftRow < $iLeftRows; $iLeftRow++) {
      $iLeftRowBase = $iLeftColumns * $iLeftRow;
      $iRightRowBase = $iRightColumns * $iLeftRow;
      // loop over cols of right
      for($iRightColumn = 0; $iRightColumn < $iRightColumns; $iRightColumn++) {

        // dot product loop
        $iDot = 0;
        //loop over columns of left
        for($iLeftColumn = 0; $iLeftColumn < $iLeftColumns; $iLeftColumn++) {
          $iRightColumnBase = $iRightColumns * $iLeftColumn;
          $iLeftIndex = $iLeftRowBase + $iLeftColumn;
          $iRightIndex = $iRightColumnBase + $iRightColumn;
          $iDot +=
              $oLeft->weights[$iLeftIndex]
            * $oRight->weights[$iRightIndex];
          $oLeft->deltas[$iLeftIndex] = 0;
          $oRight->deltas[$iRightIndex] = 0;
        }
        $oProduct->weights[$iRightRowBase + $iLeftColumn] = $iDot;
      }
    }
  }
  
  public static function fnMultiplyB(&$oProduct, &$oLeft, &$oRight) {
    $iLeftRows = $oLeft->rows;
    $iLeftColumns = $oLeft->columns;
    $iRightColumns = $oRight->columns;

    // loop over rows of left
    for($iLeftRow = 0; $iLeftRow < $iLeftRows; $iLeftRow++) {
      $iLeftRowBase = $iLeftColumns * $iLeftRow;
      $iRightRowBase = $iRightColumns * $iLeftRow;
      // loop over cols of right
      for($iRightColumn = 0; $iRightColumn < $iRightColumns; $iRightColumn++) {

        //loop over columns of left
        for($iLeftColumn = 0; $iLeftColumn < $iLeftColumns; $iLeftColumn++) {
          $iRightColumnBase = $iRightColumns * $iLeftColumn;
          $iLeftRow = $iLeftRowBase + $iLeftColumn;
          $iRightRow = $iRightColumnBase + $iRightColumn;
          $fBackPropagateValue = $oProduct->deltas[$iRightRowBase + $iRightColumn];
          $oLeft->deltas[$iLeftRow] += $oRight->weights[$iRightRow] * $fBackPropagateValue;
          $oRight->deltas[$iRightRow] += $oLeft->weights[$iLeftRow] * $fBackPropagateValue;
        }
      }
    }
  }
  
  public static function fnMultiplyElement(&$oProduct, &$oLeft, &$oRight) 
  {
    for($iI = 0; $iI < count($oLeft->weights); $iI++) {
      $oProduct->weights[$iI] = $oLeft->weights[$iI] * $oRight->weights[$iI];
      $oProduct->deltas[$iI] = 0;
    }
  }
  
  public static function fnMultiplyElementB(&$oProduct, &$oLeft, &$oRight) 
  {
    for($iI = 0; $iI < count($oLeft->weights); $iI++) {
      $oLeft->deltas[$iI] = $oRight->weights[$iI] * $oProduct->deltas[$iI];
      $oRight->deltas[$iI] = $oLeft->weights[$iI] * $oProduct->deltas[$iI];
    }
  }
  
  public static function fnRelu(&$oProduct, &$oLeft) 
  {
    for($iI = 0; $iI < count($oLeft->weights); $iI++) {
      $oProduct->weights[$iI] = max(0, $oLeft->weights[$iI]); // relu
      $oProduct->deltas[$iI] = 0;
    }
  }
  
  public static function fnReluB(&$oProduct, &$oLeft) 
  {
    for ($iI = 0; $iI < count($oProduct->deltas); $iI++) {
      $oLeft->deltas[$iI] = $oLeft->weights[$iI] > 0 ? $oProduct->deltas[$iI] : 0;
    }
  }
  
  public static function fnRowPluck(&$oProduct, &$oLeft, $iRowPluckIndex) 
  {
    $iColumns = $oLeft->columns;
    $iRowBase = $iColumns * $iRowPluckIndex;
    for ($iColumn = 0; $iColumn < $iColumns; $iColumn++) {
      $oProduct->weights[$iColumn] = $oLeft->weights[$iRowBase + $iColumn];
      $oProduct->deltas[$iColumn] = 0;
    }
  }
  
  public static function fnRowPluckB(&$oProduct, &$oLeft, $iRowIndex) 
  {
    $iColumns = $oLeft->columns;
    if (is_callable($iRowIndex)) {
      $iRowBase = $iColumns * $iRowIndex();
    } else {
      $iRowBase = $iColumns * $iRowIndex;      
    }
    for ($iColumn = 0; $iColumn < $iColumns; $iColumn++) {
      $oLeft->deltas[$iRowBase + $iColumn] = $oProduct->deltas[$iColumn];
    }
  }
  
  public static function fnSigmoid(&$oProduct, &$oLeft) 
  {
    // sigmoid nonlinearity
    for ($iI = 0; $iI < count($oLeft->weights); $iI++) {
      $oProduct->weights[$iI] = 1 / ( 1 + exp(-$oLeft->weights[$iI]));
      $oProduct->deltas[$iI] = 0;
    }
  }

  public static function fnSigmoidB(&$oProduct, &$oLeft)
  {
    for ($iI = 0; $iI < count($oProduct->deltas); $iI++) {
      $fMWI = $oProduct->weights[$iI];
      $oLeft->deltas[$iI] = $fMWI * (1 - $fMWI) * $oProduct->deltas[$iI];
    }
  }
  
  public static function fnTanh(&$oProduct, &$oLeft) 
  {
    // tanh nonlinearity
    for ($iI = 0; $iI < count($oLeft->weights); $iI++) {
      $oProduct->weights[$iI] = tanh($oLeft->weights[$iI]);
      $oProduct->deltas[$iI] = 0;
    }
  }
  
  public static function fnTanhB(&$oProduct, &$oLeft) 
  {
    for ($iI = 0; $iI < count($oProduct->deltas); $iI++) {
      // grad for z = tanh(x) is (1 - z^2)
      $fMWI = $oProduct->weights[$iI];
      $oLeft->deltas[$iI] = (1 - $fMWI * $fMWI) * $oProduct->deltas[$iI];
    }
  }
  
  public static function fnSoftmax($oM) 
  {
    $oResult = new Matrix($oM->rows, $oM->columns); // probability volume
    $iMaxVal = -999999;
    for ($iI = 0; $iI < count($oM->weights); $iI++) {
      if($oM->weights[$iI] > $iMaxVal) {
        $iMaxVal = $oM->weights[$iI];
      }
    }

    $iS = 0;
    for ($iI = 0; $iI < count($oM->weights); $iI++) {
      $oResult->weights[$iI] = exp($oM->weights[$iI] - $iMaxVal);
      $iS += $oResult->weights[$iI];
    }

    for ($iI = 0; $iI < count($oM->weights); $iI++) {
      $oResult->weights[$iI] /= $iS;
    }

    // no backward pass here needed
    // since we will use the computed probabilities outside
    // to set gradients directly on m
    return $oResult;
  }
  
  public static function fnCopy(&$oProduct, &$oLeft) 
  {
    $oProduct->rows = $oLeft->rows;
    $oProduct->columns = $oLeft->columns;
    $oProduct->weights = $oLeft->weights;
    $oProduct->deltas = $oLeft->deltas;
  }
  
  public static function fnSampleI($oM) 
  {
    // sample argmax from w, assuming w are
    // probabilities that sum to one
    $iR = Utilities::fnRandomF(0, 1);
    $iX = 0;
    $iI = 0;
    $aW = $oM->weights;

    while (true) {
      $iX += $aW[$iI];
      if($iX > $iR) {
        return $iI;
      }
      $iI++;
    }
  }
  
  public static function fnMaxI($oM) 
  {
    // argmax of array w
    $fMaxv = $oM->weights[0];
    $iMaxix = 0;
    for ($iI = 1; $iI < count($oM->weights); $iI++) {
      $fV = $oM->weights[$iI];
      if ($fV < $fMaxv) continue;

      $iMaxix = $iI;
      $fMaxv = $fV;
    }
    return $iMaxix;
  }
}
