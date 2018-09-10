<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Matrix;
use libNeuralNetwork\RandomMatrix;
use libNeuralNetwork\DataFormatter;
use Exception;
use Closure;

class RNN
{
  public $inputSize;
  public $inputRange;
  public $hiddenSizes;
  public $outputSize;
  public $learningRate;
  public $decayRate;
  public $smoothEps;
  public $regc;
  public $clipval;
  public $json;
  public $dataFormatter;
  
  public $fnSetupData;
  public $fnFormatDataIn;
  public $fnFormatDataOut;
  
  public $stepCache;
  public $runs;
  public $totalCost;
  public $ratioClipped;
  public $model;
  public $initialLayerInputs;
  public $inputLookup;
  public $outputLookup;
  
  public function fnDefaults()
  {
    return [
      'inputSize' => 20,
      'inputRange' => 20,
      'hiddenSizes' => [20,20],
      'outputSize' => 20,
      'learningRate' => 0.01,
      'decayRate' => 0.999,
      'smoothEps' => 1e-8,
      'regc' => 0.000001,
      'clipval' => 5,
      'json' => null,
      'dataFormatter' => null,        
      'fnSetupData' => function($aData) 
      {
        if (
          !is_string($aData[0])
          //&& !is_array($aData[0])
          && (
            !isset($aData[0]['input'])
            || !isset($aData[0]['output'])
          )
        ) {
          return $aData;
        }
        
        $aValues = [];
        $aResult = [];
        
        if (is_string($aData[0]) || is_array($aData[0])) {
          if ($this->dataFormatter === null) {
            for ($iI = 0; $iI < count($aData); $iI++) {
              array_push($aValues, $aData[$iI]);
            }
            $this->dataFormatter = new DataFormatter($aValues);
          }
          for ($iI = 0, $iMax = count($aData); $iI < $iMax; $iI++) {
            $fnFormatDataIn = Closure::bind($this->fnFormatDataIn, $this);
            array_push($aResult, $fnFormatDataIn($aData[$iI]));
          }
        } else {
          if ($this->dataFormatter === null) {
            for ($iI = 0; $iI < count($aData); $iI++) {
              array_push($aValues, $aData[i]['input']);
              array_push($aValues, $aData[i]['output']);
            }
            $this->dataFormatter = DataFormatter::fromArrayInputOutput($aValues);
          }
          for ($iI = 0, $iMax = count($aData); $iI < $iMax; $iI++) {
            $fnFormatDataIn = Closure::bind($this->fnFormatDataIn, $this);
            array_push($aResult, $fnFormatDataIn($aData[$iI]['input'], $aData[$iI]['output']));
          }
        }
        
        return $aResult;
      },
      'fnFormatDataIn' => function($aInput, $aOutput = null) 
      {
        if ($this->dataFormatter) {
          if (isset($this->dataFormatter->indexTable['stop-input'])) {
            return $this->dataFormatter->fnToIndexesInputOutput($aInput, $aOutput);
          } else {
            return $this->dataFormatter->fnToIndexes($aInput);
          }
        }
        return $aInput;
      },
      'fnFormatDataOut' => function($aInput, $aOutput) 
      {
        if ($this->dataFormatter) {
          return join('', $this->dataFormatter->fnToCharacters($aOutput));
        }
        return $aOutput;
      },
    ];
  }
  
  public function fnTrainDefaults()
  {
    return [
      'iterations' => 20000,
      'errorThresh' => 0.005,
      'log' => false,
      'logPeriod' => 10,
      'learningRate' => 0.3,
      'callback' => null,
      'callbackPeriod' => 10,
      'keepNetworkIntact' => false        
    ];    
  }
  
  function __construct($aOptions=[]) 
  {
    $aOptions = array_merge($this->fnDefaults(), $aOptions);

    foreach ($aOptions as $sKey => $mValue) {
      $this->$sKey = $mValue;
    }

    $this->stepCache = [];
    $this->runs = 0;
    $this->totalCost = null;
    $this->ratioClipped = null;
    $this->model = null;

    $this->initialLayerInputs = array_map(function() { return new Matrix($this->hiddenSizes[0], 1); }, $this->hiddenSizes);
    $this->inputLookup = null;
    $this->outputLookup = null;
    
    $this->fnInitialize();
  }
  
  public function fnInitialize() 
  {
    $this->model = [
      'input' => null,
      'hiddenLayers' => [],
      'output' => null,
      'equations' => [],
      'allMatrices' => [],
      'equationConnections' => []
    ];

    if ($this->dataFormatter !== null) {
      $this->inputSize =
      $this->inputRange =
      $this->outputSize = count($this->dataFormatter->characters);
    }

    if ($this->json) {
      $this->fnFromJSON($this->json);
    } else {
      $this->fnMapModel();
    }
  }
  
  public function fnCreateHiddenLayers() 
  {
    //0 is end, so add 1 to offset
    array_push($this->model['hiddenLayers'], $this->fnGetModel($this->hiddenSizes[0], $this->inputSize));
    $iPrevSize = $this->hiddenSizes[0];

    for ($iD = 1; $iD < count($this->hiddenSizes); $iD++) { // loop over depths
      $iHiddenSize = $this->hiddenSizes[$iD];
      array_push($this->model['hiddenLayers'], $this->fnGetModel($iHiddenSize, $iPrevSize));
      $iPrevSize = $iHiddenSize;
    }
  }
  
  public function fnGetModel($iHiddenSize, $iPrevSize) 
  {
    return [
      //wxh
      'weight' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //whh
      'transition' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //bhh
      'bias' => new Matrix($iHiddenSize, 1)
    ];
  }
  
  public function fnGetEquation($oEquation, $oInputMatrix, $oPreviousResult, $aHiddenLayer) 
  {
    return $oEquation->fnRelu(
      $oEquation->fnAdd(
        $oEquation->fnAdd(
          $oEquation->fnMultiply(
            $aHiddenLayer['weight'],
            $oInputMatrix
          ),
          $oEquation->fnMultiply(
            $aHiddenLayer['transition'],
            $oPreviousResult
          )
        ),
        $aHiddenLayer['bias']
      )
    );
  }

  public function fnCreateInputMatrix() 
  {
    //0 is end, so add 1 to offset
    $this->model['input'] = new RandomMatrix($this->inputRange + 1, $this->inputSize, 0.08);
  }
  
  public function fnCreateOutputMatrix() 
  {
    //0 is end, so add 1 to offset
    //whd
    $this->model['outputConnector'] = new RandomMatrix(
      $this->outputSize + 1, 
      $this->hiddenSizes[count($this->hiddenSizes) - 1], 
      0.08
    );
    //0 is end, so add 1 to offset
    //bd
    $this->model['output'] = new Matrix($this->outputSize + 1, 1);
  }
  
  public function fnBindEquation() 
  {
    $oEquation = new Equation();
    $aOutputs = [];
    $aEquationConnection = count($this->model['equationConnections']) > 0
      ? $this->model['equationConnections'][count($this->model['equationConnections']) - 1]
      : $this->initialLayerInputs
      ;

    // 0 index
    $oOutput = $this->fnGetEquation(
      $oEquation, 
      $oEquation->fnInputMatrixToRow($this->model['input']),
      $aEquationConnection[0],
      $this->model['hiddenLayers'][0]
    );
    array_push($aOutputs, $oOutput);
    // 1+ indices
    for ($iI = 1, $iMax = count($this->hiddenSizes); $iI < $iMax; $iI++) {
      $oOutput = $this->fnGetEquation($oEquation, $oOutput, $aEquationConnection[$iI], $this->model['hiddenLayers'][$iI]);
      array_push($aOutputs, $oOutput);
    }

    array_push($this->model['equationConnections'], $aOutputs);
    $oEquation->fnAdd($oEquation->fnMultiply($this->model['outputConnector'], $oOutput), $this->model['output']);
    array_push($this->model['equations'], $oEquation);
  }
  
  public function fnMapModel() 
  {
    $this->fnCreateInputMatrix();
    if (!$this->model['input']) throw new Exception('net.model.input not set');
    array_push($this->model['allMatrices'], $this->model['input']);

    $this->fnCreateHiddenLayers();
    if (!count($this->model['hiddenLayers'])) throw new Exception('net.hiddenLayers not set');
    for ($iI = 0, $iMax = count($this->model['hiddenLayers']); $iI < $iMax; $iI++) {
      foreach ($this->model['hiddenLayers'][$iI] as $mProperty) {
        array_push($this->model['allMatrices'], $mProperty);
      }
    }

    $this->fnCreateOutputMatrix();
    if (!$this->model['outputConnector']) throw new Exception('net.model.outputConnector not set');
    if (!$this->model['output']) throw new Exception('net.model.output not set');

    array_push($this->model['allMatrices'], $this->model['outputConnector']);
    array_push($this->model['allMatrices'], $this->model['output']);
  }

  public function fnTrainPattern($aInput, $iLearningRate = null) 
  {
    $iError = $this->fnRunInput($aInput);
    $this->fnRunBackpropagate($aInput);
    $this->fnStep($iLearningRate);
    return $iError;
  }

  public function fnRunInput($aInput) 
  {
    $this->runs++;
    $iMax = count($aInput);
    $iLog2ppl = 0;
    $iCost = 0;
    $oEquation;
    
    while (count($this->model['equations']) <= count($aInput) + 1) {//last is zero
      $this->fnBindEquation();
    }
    for ($iInputIndex = -1, $iInputMax = count($aInput); $iInputIndex < $iInputMax; $iInputIndex++) {
      // start and end tokens are zeros
      $iEquationIndex = $iInputIndex + 1;
      $oEquation = $this->model['equations'][$iEquationIndex];

      $iSource = ($iInputIndex === -1 ? 0 : $aInput[$iInputIndex] + 1); // first step: start with START token
      $iTarget = ($iInputIndex === $iMax - 1 ? 0 : $aInput[$iInputIndex + 1] + 1); // last step: end with END token
      $oOutput = $oEquation->fnRun($iSource);
      // set gradients into log probabilities
      $oLogProbabilities = $oOutput; // interpret output as log probabilities
      $oProbabilities = Matrix::fnSoftmax($oOutput); // compute the softmax probabilities

      $iLog2ppl += -log($oProbabilities->weights[$iTarget], 2); // accumulate base 2 log prob and do smoothing
      $iCost += -log($oProbabilities->weights[$iTarget]);
      // write gradients into log probabilities
      $oLogProbabilities->deltas = $oProbabilities->weights;
      $oLogProbabilities->deltas[$iTarget] -= 1;
    }

    $this->totalCost = $iCost;
    return pow(2, $iLog2ppl / ($iMax - 1));
  }

  public function fnRunBackpropagate($aInput) 
  {
    $iI = count($aInput);
    $aEquations = $this->model['equations'];
    while ($iI > 0) {
      $aEquations[$iI]->fnRunBackpropagate($aInput[$iI - 1] + 1);
      $iI--;
    }
    $aEquations[0]->fnRunBackpropagate(0);
  }
  
  public function fnStep($iLearningRate = null) 
  {
    // perform parameter update
    //TODO: still not sure if this is ready for learningRate
    $iStepSize = $this->learningRate;
    $iRegc = $this->regc;
    $iClipval = $this->clipval;
    $iNumClipped = 0;
    $iNumTot = 0;
    $aAllMatrices = &$this->model['allMatrices'];
    for ($iMatrixIndex = 0; $iMatrixIndex < count($aAllMatrices); $iMatrixIndex++) {
      $oMatrix = &$aAllMatrices[$iMatrixIndex];
      if (!isset($this->stepCache[$iMatrixIndex])) {
        $this->stepCache[$iMatrixIndex] = Utilities::fnZeros($oMatrix->rows * $oMatrix->columns);
      }
      $aCache = $this->stepCache[$iMatrixIndex];
      for ($iI = 0; $iI < count($oMatrix->weights); $iI++) {
        $iR = $oMatrix->deltas[$iI];
        $iW = $oMatrix->weights[$iI];
        // rmsprop adaptive learning rate
        $aCache[$iI] = $aCache[$iI] * $this->decayRate + (1 - $this->decayRate) * $iR * $iR;
        // gradient clip
        if ($iR > $iClipval) {
          $iR = $iClipval;
          $iNumClipped++;
        }
        if ($iR < -$iClipval) {
          $iR = -$iClipval;
          $iNumClipped++;
        }
        $iNumTot++;
        // update (and regularize)
        $oMatrix->weights[$iI] = $iW + -$iStepSize * $iR / sqrt($aCache[$iI] + $this->smoothEps) - $iRegc * $iW;
      }
    }
    $this->ratioClipped = $iNumClipped / $iNumTot;
  }
  
  public function fnIsRunnable()
  {
    if(count($this->model['equations']) === 0){
      echo (`No equations bound, did you run train()?`);
      return false;
    }

    return true;
  }
  
  public function fnRun($aRawInput = [], $iMaxPredictionLength = 100, $bIsSampleI = false, $iTemperature = 1) 
  {
    if (!$this->fnIsRunnable()) 
      return null;

    $fnFormatDataIn = Closure::bind($this->fnFormatDataIn, $this);
    $aInput = $fnFormatDataIn($aRawInput);
    $aOutput = [];
    $iI = 0;
    while (count($this->model['equations']) < $iMaxPredictionLength) {
      $this->fnBindEquation();
    }
    while (true) {
      $iPreviousIndex = ($iI === 0
        ? 0
        : $iI < count($aInput)
          ? $aInput[$iI - 1] + 1
          : $aOutput[$iI - 1])
          ;
      $oEquation = $this->model['equations'][$iI];
      // sample predicted letter
      $oOutputMatrix = $oEquation->fnRun($iPreviousIndex);
      $oLogProbabilities = new Matrix($this->model['output']->rows, $this->model['output']->columns);
      Matrix::fnCopy($oLogProbabilities, $oOutputMatrix);
      if ($iTemperature !== 1 && $bIsSampleI) {
        /**
         * scale log probabilities by temperature and re-normalize
         * if temperature is high, logProbabilities will go towards zero
         * and the softmax outputs will be more diffuse. if temperature is
         * very low, the softmax outputs will be more peaky
         */
        for ($iJ = 0, $iMax = count($oLogProbabilities->weights); $iJ < $iMax; $iJ++) {
          $oLogProbabilities->weights[$iJ] /= $iTemperature;
        }
      }

      $oProbs = Matrix::fnSoftmax($oLogProbabilities);
      $iNextIndex = ($bIsSampleI ? Matrix::fnSampleI($oProbs) : Matrix::fnMaxI($oProbs));

      $iI++;
      if ($iNextIndex === 0) {
        // END token predicted, break out
        break;
      }
      if ($iI >= $iMaxPredictionLength) {
        // something is wrong
        break;
      }

      array_push($aOutput, $iNextIndex);
    }

    /**
     * we slice the input length here, not because output contains it, but it will be erroneous as we are sending the
     * network what is contained in input, so the data is essentially guessed by the network what could be next, till it
     * locks in on a value.
     * Kind of like this, values are from input:
     * 0 -> 4 (or in English: "beginning on input" -> "I have no idea? I'll guess what they want next!")
     * 2 -> 2 (oh how interesting, I've narrowed down values...)
     * 1 -> 9 (oh how interesting, I've now know what the values are...)
     * then the output looks like: [4, 2, 9,...]
     * so we then remove the erroneous data to get our true output
     */
    $fnFormatDataOut = Closure::bind($this->fnFormatDataOut, $this);
    return $fnFormatDataOut(
      $aInput,
      array_map(function($v) { return $v - 1; }, array_slice($aOutput, 0, count($aInput)))
    );
  }

  public function fnTrain($aData, $aOptions = []) 
  {
    $aOptions = array_merge($this->fnTrainDefaults(), $aOptions);
    $iIterations = $aOptions['iterations'];
    $iErrorThresh = $aOptions['errorThresh'];
    $oLog = $aOptions['log'] === true ? function($v) { echo $v."\n"; } : $aOptions['log'];
    $iLogPeriod = $aOptions['logPeriod'];
    $iLearningRate = $aOptions['learningRate'] || $this->learningRate;
    $oCallback = $aOptions['callback'];
    $iCallbackPeriod = $aOptions['callbackPeriod'];
    $iError = INF;
    $iI;

    if ($this->fnSetupData) {
      $fnSetupData = Closure::bind($this->fnSetupData, $this);
      $aData = $fnSetupData($aData);
    }

    if (!$aOptions['keepNetworkIntact']) {
      $this->fnInitialize();
    }

    for ($iI = 0; $iI < $iIterations && $iError > $iErrorThresh; $iI++) {
      $iSum = 0;
      for ($iJ = 0; $iJ < count($aData); $iJ++) {
        $iErr = $this->fnTrainPattern($aData[$iJ], $iLearningRate);
        $iSum += $iErr;
      }
      $iError = ($iSum / count($aData));

      if ($iError == INF) 
        throw new Exception('network error rate is unexpected NaN, check network configurations and try again');
      
      if ($oLog && ($iI % $iLogPeriod == 0)) {
        $oLog('iterations: '. $iI. ' training error: '. $iError);
      }
      if ($oCallback && ($iI % $iCallbackPeriod == 0)) {
        $oCallback([ 'error' => $iError, 'iterations' => $iI ]);
      }
    }

    return [
      'error' => $iError,
      'iterations' => $iI
    ];
  }

  public function fnTest($aData) {
    throw new Exception('not yet implemented');
  }
}
