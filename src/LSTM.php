<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Matrix;
use libNeuralNetwork\RandomMatrix;
use libNeuralNetwork\RNN;

class LSTM extends RNN
{
  public function fnGetModel($iHiddenSize, $iPrevSize) 
  {
    return [
      // gates parameters
      //wix
      'inputMatrix' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //wih
      'inputHidden' => new RandomMatrix($iHiddenSize, $iHiddenSize, 0.08),
      //bi
      'inputBias' => new Matrix($iHiddenSize, 1),

      //wfx
      'forgetMatrix' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //wfh
      'forgetHidden' => new RandomMatrix($iHiddenSize, $iHiddenSize, 0.08),
      //bf
      'forgetBias' => new Matrix($iHiddenSize, 1),

      //wox
      'outputMatrix' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //woh
      'outputHidden' => new RandomMatrix($iHiddenSize, $iHiddenSize, 0.08),
      //bo
      'outputBias' => new Matrix($iHiddenSize, 1),

      // cell write params
      //wcx
      'cellActivationMatrix' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //wch
      'cellActivationHidden' => new RandomMatrix($iHiddenSize, $iHiddenSize, 0.08),
      //bc
      'cellActivationBias' => new Matrix($iHiddenSize, 1)
    ];
  }
  
  public function fnGetEquation($oEquation, $oInputMatrix, $oPreviousResult, $aHiddenLayer) 
  {
    $oInputGate = $oEquation->fnSigmoid(
      $oEquation->fnAdd(
        $oEquation->fnAdd(
          $oEquation->fnMultiply(
            $aHiddenLayer['inputMatrix'],
            $oInputMatrix
          ),
          $oEquation->fnMultiply(
            $aHiddenLayer['inputHidden'],
            $oPreviousResult
          )
        ),
        $aHiddenLayer['inputBias']
      )
    );

    $oForgetGate = $oEquation->fnSigmoid(
      $oEquation->fnAdd(
        $oEquation->fnAdd(
          $oEquation->fnMultiply(
            $aHiddenLayer['forgetMatrix'],
            $oInputMatrix
          ),
          $oEquation->fnMultiply(
            $aHiddenLayer['forgetHidden'],
            $oPreviousResult
          )
        ),
        $aHiddenLayer['forgetBias']
      )
    );

    // output gate
    $oOutputGate = $oEquation->fnSigmoid(
      $oEquation->fnAdd(
        $oEquation->fnAdd(
          $oEquation->fnMultiply(
            $aHiddenLayer['outputMatrix'],
            $oInputMatrix
          ),
          $oEquation->fnMultiply(
            $aHiddenLayer['outputHidden'],
            $oPreviousResult
          )
        ),
        hiddenLayer['outputBias']
      )
    );

    // write operation on cells
    $oCellWrite = $oEquation->fnTanh(
      $oEquation->fnAdd(
        $oEquation->fnAdd(
          $oEquation->fnMultiply(
            $aHiddenLayer['cellActivationMatrix'],
            $oInputMatrix
          ),
          $oEquation->fnMultiply(
            $aHiddenLayer['cellActivationHidden'],
            $oPreviousResult
          )
        ),
        $aHiddenLayer['cellActivationBias']
      )
    );

    // compute new cell activation
    $oRetainCell = $oEquation->fnMultiplyElement($oForgetGate, $oPreviousResult); // what do we keep from cell
    $oWriteCell = $oEquation->fnMultiplyElement($oInputGate, $oCellWrite); // what do we write to cell
    $oCell = $oEquation->fnAdd($oRetainCell, $oWriteCell); // new cell contents

    // compute hidden state as gated, saturated cell activations
    return $oEquation->fnMultiplyElement(
      $oOutputGate,
      $oEquation->fnTanh($oCell)
    );
  }
}

