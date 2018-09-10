<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Lookup;
use libNeuralNetwork\Utilities;
use Exception;
use Closure;

class NeuralNetwork 
{
  public $outputLookup;
  public $hiddenSizes;
  public $trainOpts;
  public $sizes;
  public $outputLayer;
  public $biases;
  public $weights;
  public $outputs;
  public $deltas;
  public $changes;
  public $errors;
  public $errorCheckInterval;
  public $fnRunInput;
  public $fnCalculateDeltas;
  public $activation;
  public $binaryThresh;
  public $hiddenLayers;
  public $fnRunInputSigmoid;
  public $fnRunInputRelu;
  public $fnRunInputLeakyRelu;
  public $fnRunInputTanh;
  public $fnCalculateDeltasSigmoid;
  public $fnCalculateDeltasRelu;
  public $fnCalculateLeakyRelu;
  public $fnCalculateTanh;
          
  public static function fnTrainDefaults()
  {
    return [
      "iterations" => 20000,    // the maximum times to iterate the training data
      "errorThresh" => 0.005,   // the acceptable error percentage from training data
      "log" => false,           // true to use console.log, when a function is supplied it is used
      "logPeriod" => 10,        // iterations between logging out
      "learningRate" => 0.3,    // multiply's against the input and the delta then adds to momentum
      "momentum" => 0.1,        // multiply's against the specified "change" then adds to learning rate for change
      "callback" => null,       // a periodic call back that can be triggered while training
      "callbackPeriod" => 10,   // the number of iterations through the training data between callback calls
      "timeout" => INF          // the max number of milliseconds to train for
    ];
  }
  
  public static function fnDefaults() 
  {
    return [
      "binaryThresh" => 0.5,     // ¯\_(ツ)_/¯
      "hiddenLayers" => [3],     // array of ints for the sizes of the hidden layers in the network
      "activation" => 'sigmoid'  // Supported activation types ['sigmoid', 'relu', 'leaky-relu', 'tanh']
    ];
  }
  
  public static function fnValidateTrainingOptions($aOptions) 
  {
    $aValidations = [
      "iterations" => function ($v) { return is_numeric($v) && $v > 0; },
      "errorThresh" => function ($v) { return is_numeric($v) && $v > 0 && $v < 1; },
      "log" => function ($v) { return is_callable($v) || is_bool($v); },
      "logPeriod" => function ($v) { return is_numeric($v) && $v > 0; },
      "learningRate" => function ($v) { return is_numeric($v) && $v > 0 && $v < 1; },
      "momentum" => function ($v) { return is_numeric($v) && $v > 0 && $v < 1; },
      "callback" => function ($v) { return is_callable($v) || $v === null; },
      "callbackPeriod" => function ($v) { return is_numeric($v) && $v > 0; },
      "timeout" => function ($v) { return is_numeric($v) && $v > 0; }
    ];
    foreach (self::fnTrainDefaults() as $sKey => $mValue) {
      if (isset($aValidations[$sKey]) && !$aValidations[$sKey]($aOptions[$sKey])) {
        throw new Exception("[$sKey, {$aOptions[$sKey]}] is out of normal training range, your network will probably not train.");
      }
    }
  }
  
  function __construct($aOptions = []) 
  {
    $aOptions = array_merge(self::fnDefaults(), $aOptions);
    foreach ($aOptions as $sKey => $mValue) {
      $this->$sKey = $mValue;
    }
    
    $this->hiddenSizes = $aOptions["hiddenLayers"];
    $this->trainOpts = self::fnTrainDefaults();
    $this->fnUpdateTrainingOptions(array_merge(self::fnTrainDefaults(), $aOptions));

    $this->sizes = null;
    $this->outputLayer = null;
    $this->biases = null; // weights for bias nodes
    $this->weights = null;
    $this->outputs = null;

    // state for training
    $this->deltas = null;
    $this->changes = null; // for momentum
    $this->errors = null;
    $this->errorCheckInterval = 1;
    
    //if (!$this->constructor.prototype.hasOwnProperty('runInput')) {
      $this->fnRunInput = null;
    //}
    //if (!$this->constructor.prototype.hasOwnProperty('calculateDeltas')) {
      $this->fnCalculateDeltas = null;
    //}
      
    $this->fnRunInputSigmoid = function($aInput) 
    {
      $this->outputs[0] = $aInput;  // set output state of input layer

      $aOutput = null;

      for ($iLayer = 1; $iLayer <= $this->outputLayer; $iLayer++) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $aWeights = $this->weights[$iLayer][$iNode];

          $iSum = $this->biases[$iLayer][$iNode];
          for ($iK = 0; $iK < count($aWeights); $iK++) {
            $iSum += $aWeights[$iK] * $aInput[$iK];
          }
          //sigmoid
          $this->outputs[$iLayer][$iNode] = 1 / (1 + exp(-$iSum));
        }
        $aOutput = $aInput = $this->outputs[$iLayer];
      }

      return $aOutput;
    };
  
    $this->fnRunInputRelu = function($aInput) 
    {
      $this->outputs[0] = $aInput;  // set output state of input layer

      $aOutput = null;

      for ($iLayer = 1; $iLayer <= $this->outputLayer; $iLayer++) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $aWeights = $this->weights[$iLayer][$iNode];

          $iSum = $this->biases[$iLayer][$iNode];
          for ($iK = 0; $iK < count($aWeights); $iK++) {
            $iSum += $aWeights[$iK] * $aInput[$iK];
          }
          //relu
          $this->outputs[$iLayer][$iNode] = ($iSum < 0 ? 0 : $iSum);
        }
        $aOutput = $aInput = $this->outputs[$iLayer];
      }

      return $aOutput;
    };
  
    $this->fnRunInputLeakyRelu = function($aInput) 
    {
      $this->outputs[0] = $aInput;  // set output state of input layer

      $aOutput = null;

      for ($iLayer = 1; $iLayer <= $this->outputLayer; $iLayer++) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $aWeights = $this->weights[$iLayer][$iNode];

          $iSum = $this->biases[$iLayer][$iNode];
          for ($iK = 0; $iK < count($aWeights); $iK++) {
            $iSum += $aWeights[$iK] * $aInput[$iK];
          }
          //relu
          $this->outputs[$iLayer][$iNode] = ($iSum < 0 ? 0 : 0.01 * $iSum);
        }
        $aOutput = $aInput = $this->outputs[$iLayer];
      }

      return $aOutput;
    };
  
    $this->fnRunInputTanh = function($aInput) 
    {
      $this->outputs[0] = $aInput;  // set output state of input layer

      $aOutput = null;

      for ($iLayer = 1; $iLayer <= $this->outputLayer; $iLayer++) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $aWeights = $this->weights[$iLayer][$iNode];

          $iSum = $this->biases[$iLayer][$iNode];
          for ($iK = 0; $iK < count($aWeights); $iK++) {
            $iSum += $aWeights[$iK] * $aInput[$iK];
          }
          //tanh
          $this->outputs[$iLayer][$iNode] = tanh($iSum);
        }
        $aOutput = $aInput = $this->outputs[$iLayer];
      }

      return $aOutput;
    };
    
    $this->fnCalculateDeltasSigmoid = function($aTarget) 
    {
      for ($iLayer = $this->outputLayer; $iLayer >= 0; $iLayer--) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $iOutput = $this->outputs[$iLayer][$iNode];

          $iError = 0;
          if ($iLayer === $this->outputLayer) {
            $iError = $aTarget[$iNode] - $iOutput;
          } else {
            $aDeltas = $this->deltas[$iLayer + 1];
            for ($iK = 0; $iK < count($aDeltas); $iK++) {
              $iError += $aDeltas[$iK] * $this->weights[$iLayer + 1][$iK][$iNode];
            }
          }
          $this->errors[$iLayer][$iNode] = $iError;
          $this->deltas[$iLayer][$iNode] = $iError * $iOutput * (1 - $iOutput);
        }
      }
    };

    $this->fnCalculateDeltasRelu = function($aTarget) 
    {
      for ($iLayer = $this->outputLayer; $iLayer >= 0; $iLayer--) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $iOutput = $this->outputs[$iLayer][$iNode];

          $iError = 0;
          if ($iLayer === $this->outputLayer) {
            $iError = $aTarget[$iNode] - $iOutput;
          } else {
            $aDeltas = $this->deltas[$iLayer + 1];
            for ($iK = 0; $iK < count($aDeltas); $iK++) {
              $iError += $aDeltas[$iK] * $this->weights[$iLayer + 1][$iK][$iNode];
            }
          }
          $this->errors[$iLayer][$iNode] = $iError;
          $this->deltas[$iLayer][$iNode] = $iOutput > 0 ? $iError : 0;
        }
      }
    };

    $this->fnCalculateDeltasLeakyRelu = function($aTarget) 
    {
      for ($iLayer = $this->outputLayer; $iLayer >= 0; $iLayer--) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $iOutput = $this->outputs[$iLayer][$iNode];

          $iError = 0;
          if ($iLayer === $this->outputLayer) {
            $iError = $aTarget[$iNode] - $iOutput;
          } else {
            $aDeltas = $this->deltas[$iLayer + 1];
            for ($iK = 0; $iK < count($aDeltas); $iK++) {
              $iError += $aDeltas[$iK] * $this->weights[$iLayer + 1][$iK][$iNode];
            }
          }
          $this->errors[$iLayer][$iNode] = $iError;
          $this->deltas[$iLayer][$iNode] = $iOutput > 0 ? $iError : 0.01 * $iError;
        }
      }
    };

    $this->fnCalculateDeltasTanh = function($aTarget) 
    {
      for ($iLayer = $this->outputLayer; $iLayer >= 0; $iLayer--) {
        for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
          $iOutput = $this->outputs[$iLayer][$iNode];

          $iError = 0;
          if ($iLayer === $this->outputLayer) {
            $iError = $aTarget[$iNode] - $iOutput;
          } else {
            $aDeltas = $this->deltas[$iLayer + 1];
            for ($iK = 0; $iK < count($aDeltas); $iK++) {
              $iError += $aDeltas[$iK] * $this->weights[$iLayer + 1][$iK][$iNode];
            }
          }
          $this->errors[$iLayer][$iNode] = $iError;
          $this->deltas[$iLayer][$iNode] = (1 - $iOutput * $iOutput) * $iError;
        }
      }
    };
  }
  
  public function fnInitialize() 
  {
    if (!$this->sizes) 
      throw new Exception('Sizes must be set before initializing');

    $this->outputLayer = count($this->sizes) - 1;
    $this->biases = []; // weights for bias nodes
    $this->weights = [];
    $this->outputs = [];

    // state for training
    $this->deltas = [];
    $this->changes = []; // for momentum
    $this->errors = [];

    for ($iLayer = 0; $iLayer <= $this->outputLayer; $iLayer++) {
      $iSize = $this->sizes[$iLayer];
      $this->deltas[$iLayer] = Utilities::fnZeros($iSize);
      $this->errors[$iLayer] = Utilities::fnZeros($iSize);
      $this->outputs[$iLayer] = Utilities::fnZeros($iSize);

      if ($iLayer > 0) {
        $this->biases[$iLayer] = Utilities::fnRandos($iSize);
        $this->weights[$iLayer] = array_fill(0, $iSize, 0);
        $this->changes[$iLayer] = array_fill(0, $iSize, 0);

        for ($iNode = 0; $iNode < $iSize; $iNode++) {
          $iPrevSize = $this->sizes[$iLayer - 1];
          $this->weights[$iLayer][$iNode] = Utilities::fnRandos($iPrevSize);
          $this->changes[$iLayer][$iNode] = Utilities::fnZeros($iPrevSize);
        }
      }
    }

    $this->fnSetActivation();
  }

  public function fnSetActivation($sActivation=null) 
  {
    $this->activation = ($sActivation) ? $sActivation : $this->activation;
    switch ($this->activation) {
      case 'sigmoid':
        $this->fnRunInput = !empty($this->fnRunInput) ? $this->fnRunInput : $this->fnRunInputSigmoid;
        $this->fnCalculateDeltas = !empty($this->fnCalculateDeltas) ? $this->fnCalculateDeltas : $this->fnCalculateDeltasSigmoid;
        break;
      case 'relu':
        $this->fnRunInput = !empty($this->fnRunInput) ? $this->fnRunInput : $this->fnRunInputRelu;
        $this->fnCalculateDeltas = !empty($this->fnCalculateDeltas) ? $this->fnCalculateDeltas : $this->fnCalculateDeltasRelu;
        break;
      case 'leaky-relu':
        $this->fnRunInput = !empty($this->fnRunInput) ? $this->fnRunInput : $this->fnRunInputLeakyRelu;
        $this->fnCalculateDeltas = !empty($this->fnCalculateDeltas) ? $this->fnCalculateDeltas : $this->fnCalculateDeltasLeakyRelu;
        break;
      case 'tanh':
        $this->fnRunInput = !empty($this->fnRunInput) ? $this->fnRunInput : $this->fnRunInputTanh;
        $this->fnCalculateDeltas = !empty($this->fnCalculateDeltas) ? $this->fnCalculateDeltas : $this->fnCalculateDeltasTanh;
        break;
      default:
        throw new Exception('unknown activation '.$this->activation.', The activation should be one of [\'sigmoid\', \'relu\', \'leaky-relu\', \'tanh\']');
    }
  }
  
  public function fnIsRunnable()
  {
    if(!$this->fnRunInput){
      echo ('Activation function has not been initialized, did you run train()?');
      return false;
    }

    $checkFns = [
      'sizes',
      'outputLayer',
      'biases',
      'weights',
      'outputs',
      'deltas',
      'changes',
      'errors',
    ];
    
    $checkFns = array_filter($checkFns, function($v) { return $this->$v === null; });

    if(count($checkFns) > 0){
      echo ("Some settings have not been initialized correctly, did you run train()? Found issues with: ".join(",", $checkFns));
      return false;
    }
    return true;
  }
  
  public function fnRun($aInput) 
  {
    //if (!$this->isRunnable) return null;
    if ($this->inputLookup) {
      $aInput = Lookup::fnToArray($this->inputLookup, $aInput);
    }

    $fnRunInput = Closure::bind($this->fnRunInput, $this);
    $aOutput = $fnRunInput($aInput);

    if ($this->outputLookup) {
      $aOutput = Lookup::fnToHash($this->outputLookup, $aOutput);
    }
    return $aOutput;
  }
    
  public function fnVerifyIsInitialized($aData) 
  {
    if ($this->sizes) return;

    $this->sizes = [];
    array_push($this->sizes, count($aData[0]['input']));
    if (!$this->hiddenSizes) {
      array_push($this->sizes, max(3, floor(count($aData[0]['input']) / 2)));
    } else {
      foreach ($this->hiddenSizes as $iSize) {
        array_push($this->sizes, $iSize);
      }
    }
    array_push($this->sizes, count($aData[0]['output']));

    $this->fnInitialize();
  }
  
  public function fnUpdateTrainingOptions($aOptions) 
  {
    foreach (self::fnTrainDefaults() as $sKey => $mValue) {
      $this->trainOpts[$sKey] = isset($aOptions[$sKey]) ? $aOptions[$sKey] : $this->trainOpts[$sKey];
    }
    
    self::fnValidateTrainingOptions($this->trainOpts);
    
    $this->fnSetLogMethod(isset($aOptions['log']) ? $aOptions['log'] : $this->trainOpts['log']);
    $this->activation = isset($aOptions['activation']) ? $aOptions['activation'] : $this->activation;
  }
  
  public function getTrainOptsJSON()
  {
    return array_reduce(
      array_keys(self::fnTrainDefaults()),
      function ($aOptions, $sOption) {
        if ($sOption === 'timeout' && $this->trainOpts[$sOption] === -1) 
          return $aOptions;
        if ($this->trainOpts[$sOption]) 
          $aOptions[$sOption] = $this->trainOpts[$sOption];
        if ($sOption === 'log') 
          $aOptions['log'] = is_callable($aOptions['log']);
        return $aOptions;
      },
      []
    );
  }
  
  public function fnSetLogMethod($mLog) 
  {
    if (is_callable($mLog)){
      $this->trainOpts['log'] = $mLog;
    } else if ($mLog) {
      $this->trainOpts['log'] = function($v) { echo $v."\n"; };
    } else {
      $this->trainOpts['log'] = false;
    }
  }
  
  public function fnCalculateTrainingError($aData) 
  {
    $iSum = 0;
    
    for ($iI = 0; $iI < count($aData); ++$iI) {
      $iSum += $this->fnTrainPattern($aData[$iI]['input'], $aData[$iI]['output'], true);
    }
    
    return $iSum / count($aData);
  }
  
  public function fnTrainPatterns($aData) 
  {
    for ($iI = 0; $iI < count($aData); ++$iI) {
      $this->fnTrainPattern($aData[$iI]['input'], $aData[$iI]['output'], false);
    }
  }
  
  public function fnTrainingTick($aData, &$aStatus, $iEndTime) 
  {
    if ($aStatus['iterations'] >= $this->trainOpts['iterations'] || $aStatus['error'] <= $this->trainOpts['errorThresh'] || time() >= $iEndTime) {
      return false;
    }

    $aStatus['iterations']++;
    
    if ($this->trainOpts['log'] && ($aStatus['iterations'] % $this->trainOpts['logPeriod'] === 0)) {
      $aStatus['error'] = $this->fnCalculateTrainingError($aData);
      $this->trainOpts['log']("iterations: {$aStatus['iterations']}, training error: {$aStatus['error']}");
    } else {
      if ($aStatus['iterations'] % $this->errorCheckInterval === 0) {
        $aStatus['error'] = $this->fnCalculateTrainingError($aData);
      } else {
        $this->fnTrainPatterns($aData);
      }
    }

    if ($this->trainOpts['callback'] && ($aStatus['iterations'] % $this->trainOpts['callbackPeriod'] === 0)) {
      $this->trainOpts['callback']($aStatus);
    }
    return true;
  }
  
  public function fnPrepTraining($aData, $aOptions) 
  {
    $this->fnUpdateTrainingOptions($aOptions);
    $aData = $this->fnFormatData($aData);
    $iEndTime = time() + $this->trainOpts['timeout'];

    $aStatus = [
      "error" => 1,
      "iterations" => 0
    ];

    $this->fnVerifyIsInitialized($aData);

    return [
      "aData" => $aData,
      "aStatus" => $aStatus,
      "iEndTime" => $iEndTime
    ];
  }
  
  public function fnTrain($aData, $aOptions = []) 
  {
    extract($this->fnPrepTraining($aData, $aOptions));

    while ($this->fnTrainingTick($aData, $aStatus, $iEndTime));
    return $aStatus;
  }
  
  public function fnTrainPattern($aInput, $aTarget, $blogErrorRate) 
  {
    // forward propagate
    $fnRunInput = Closure::bind($this->fnRunInput, $this);
    $fnRunInput($aInput);

    // back propagate
    $fnCalculateDeltas = Closure::bind($this->fnCalculateDeltas, $this);
    $fnCalculateDeltas($aTarget);
    $this->fnAdjustWeights();

    if ($blogErrorRate) {
      return Utilities::fnMse($this->errors[$this->outputLayer]);
    } else {
      return null;
    }
  }
  
  public function fnAdjustWeights() 
  {
    for ($iLayer = 1; $iLayer <= $this->outputLayer; $iLayer++) {
      $iIncoming = $this->outputs[$iLayer - 1];

      for ($iNode = 0; $iNode < $this->sizes[$iLayer]; $iNode++) {
        $iDelta = $this->deltas[$iLayer][$iNode];

        for ($iK = 0; $iK < count($iIncoming); $iK++) {
          $iChange = $this->changes[$iLayer][$iNode][$iK];

          $iChange = ($this->trainOpts['learningRate'] * $iDelta * $iIncoming[$iK])
            + ($this->trainOpts['momentum'] * $iChange);

          $this->changes[$iLayer][$iNode][$iK] = $iChange;
          $this->weights[$iLayer][$iNode][$iK] += $iChange;
        }
        $this->biases[$iLayer][$iNode] += $this->trainOpts['learningRate'] * $iDelta;
      }
    }
  }
  
  public function fnFormatData($aData) 
  {
    if (!is_array($aData)) { // turn stream datum into array
      $aData = [$aData];
    }
    // turn sparse hash input into arrays with 0s as filler
    $mDatum = $aData[0]['input'];
    if (is_array($mDatum)) {
      if (!$this->inputLookup) {
        $this->inputLookup = Lookup::fnBuildLookup(
          array_map(function($v) { return $v['input']; }, $aData)
        );
      }
      $aData = array_map(
        function($v) 
        {
          $aArray = Lookup::fnToArray($this->inputLookup, $v['input']);
          return array_merge($v, [ 'input' => $aArray ]);
        },
        $aData
      );
    }

    if (is_array($aData[0]['output'])) {
      if (!$this->outputLookup) {
        $this->outputLookup = Lookup::fnBuildLookup(
          array_map(function($v) { return $v['output']; }, $aData)
        );
      }
      $aData = array_map(
        function($v) 
        {
          $aArray = Lookup::fnToArray($this->outputLookup, $v['output']);
          return array_merge($v, [ 'output' => $aArray ]);
        },
        $aData
      );
    }
    
    return $aData;
  }
  
  public function fnTest($aData) 
  {
    $aData = $this->fnFormatData($aData);

    // for binary classification problems with one output node
    $bIsBinary = count($aData[0]['output']) === 1;
    $iFalsePos = 0;
    $iFalseNeg = 0;
    $iTruePos = 0;
    $iTrueNeg = 0;

    // for classification problems
    $aMisclasses = [];

    // run each pattern through the trained network and collect
    // error and misclassification statistics
    $iSum = 0;
    for ($iI = 0; $iI < count($aData); $iI++) {
      $fnRunInput = Closure::bind($this->fnRunInput, $this);
      $aOutput = $fnRunInput($aData[0]['input']);
      $aTarget = $aData[$iI]['output'];

      $iActual = null;
      $iExpected = null;
      
      if ($bIsBinary) {
        $iActual = $aOutput[0] > $this->binaryThresh ? 1 : 0;
        $iExpected = $aTarget[0];
      }
      else {
        $iActual = array_search(Utilities::fnMax($aOutput), $aOutput);
        $iExpected = array_search(Utilities::fnMax($aTarget), $aTarget);
      }

      if ($iActual !== $iExpected) {
        $iMisclass = $aData[$iI];
        $iMisclass = array_merge(
          $iMisclass, 
          [
            'actual' => $iActual,
            'expected' => $iExpected
          ]
        );
        array_push($aMisclasses, $iMisclass);
      }

      if ($bIsBinary) {
        if ($iActual === 0 && $iExpected === 0) {
          $iTrueNeg++;
        } else if ($iActual === 1 && $iExpected === 1) {
          $iTruePos++;
        } else if ($iActual === 0 && $iExpected === 1) {
          $iFalseNeg++;
        } else if ($iActual === 1 && $iExpected === 0) {
          $iFalsePos++;
        }
      }

      $aErrors = [];
      foreach ($aOutput as $iKey => $iValue) {
        $aErrors[] = $aTarget[$iKey] - $iValue;
      }
      $iSum += Utilities::fnMse($aErrors);
    }
    $fError = $iSum / count($aData);

    $aStats = [
      'error' => $fError,
      'misclasses' => $aMisclasses
    ];

    if ($bIsBinary) {
      $aStats = array_merge(
        $aStats,
        [
          'trueNeg' => $iTrueNeg,
          'truePos' => $iTruePos,
          'falseNeg' => $iFalseNeg,
          'falsePos' => $iFalsePos,
          'total' => count($aData),
          'precision' => $iTruePos / ($iTruePos + $iFalsePos),
          'recall' => $iTruePos / ($iTruePos + $iFalseNeg),
          'accuracy' => ($iTrueNeg + $iTruePos) / count($aData)
        ]
      );
    }
    return $aStats;
  }
  
  public function fnCreateTrainStream($aOptions = []) 
  {
    $aOptions['neuralNetwork'] = &$this;
    $this->setActivation();
    //$this->trainStream = new TrainStream($aOptions);
    return $this->trainStream;
  }
}
