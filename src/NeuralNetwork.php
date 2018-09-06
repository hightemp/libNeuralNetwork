<?php

namespace libNeuralNetwork;

class NeuralNetwork 
{
  protected $hiddenSizes;
  protected $trainOpts;
  protected $sizes;
  protected $outputLayer;
  protected $biases;
  protected $weights;
  protected $outputs;
  protected $deltas;
  protected $changes;
  protected $errors;
  protected $errorCheckInterval;
  protected $activation;

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
      "timeout" => -1           // the max number of milliseconds to train for
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
    //Object.assign(this, self::fnDefaults(), $aOptions);
    $this->hiddenSizes = $aOptions["hiddenLayers"];
    $this->trainOpts = [];
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
      $this->runInput = null;
    //}
    //if (!$this->constructor.prototype.hasOwnProperty('calculateDeltas')) {
      $this->calculateDeltas = null;
    //}
  }
  
  public function fnInitialize() 
  {
    if (!$this->sizes) 
      throw new Exception('Sizes must be set before initializing');

    $this->outputLayer = $this->sizes.length - 1;
    $this->biases = []; // weights for bias nodes
    $this->weights = [];
    $this->outputs = [];

    // state for training
    $this->deltas = [];
    $this->changes = []; // for momentum
    $this->errors = [];

    for ($iLayer = 0; $iLayer <= $this->outputLayer; $iLayer++) {
      $iSize = $this->sizes[$iLayer];
      $this->deltas[$iLayer] = zeros($iSize);
      $this->errors[$iLayer] = zeros($iSize);
      $this->outputs[$iLayer] = zeros($iSize);

      if ($iLayer > 0) {
        $this->biases[$iLayer] = randos($iSize);
        $this->weights[$iLayer] = array_fill(0, $iSize, 0);
        $this->changes[$iLayer] = array_fill(0, $iSize, 0);

        for ($iNode = 0; $iNode < $iSize; $iNode++) {
          $iPrevSize = $this->sizes[$iLayer - 1];
          $this->weights[$iLayer][$iNode] = randos($iPrevSize);
          $this->changes[$iLayer][$iNode] = zeros($iPrevSize);
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
        $this->runInput = $this->runInput || $this->_runInputSigmoid;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasSigmoid;
        break;
      case 'relu':
        $this->runInput = $this->runInput || $this->_runInputRelu;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasRelu;
        break;
      case 'leaky-relu':
        $this->runInput = $this->runInput || $this->_runInputLeakyRelu;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasLeakyRelu;
        break;
      case 'tanh':
        $this->runInput = $this->runInput || $this->_runInputTanh;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasTanh;
        break;
      default:
        throw new Exception('unknown activation '.$this->activation.', The activation should be one of [\'sigmoid\', \'relu\', \'leaky-relu\', \'tanh\']');
    }
  }
  
  public function fnIsRunnable()
  {
    if(!$this->runInput){
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

    $aOutput = $this->runInput($aInput);

    if ($this->outputLookup) {
      $aOutput = Lookup::fnToHash($this->outputLookup, $aOutput);
    }
    return $aOutput;
  }
  
  public function fnRunInputSigmoid($aInput) 
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
        $this->outputs[$iLayer][$iNode] = 1 / (1 + Math.exp(-$iSum));
      }
      $aOutput = $aInput = $this->outputs[$iLayer];
    }
    
    return $aOutput;
  }
  
  public function fnRunInputRelu($aInput) 
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
  }
  
  public function fnRunInputLeakyRelu($aInput) 
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
  }
  
  public function fnRunInputTanh($aInput) 
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
        $this->outputs[$iLayer][$iNode] = Math.tanh($iSum);
      }
      $aOutput = $aInput = $this->outputs[$iLayer];
    }
    
    return $aOutput;
  }
  
  public function fnVerifyIsInitialized($aData) 
  {
    if ($this->sizes) return;

    $this->sizes = [];
    array_push($this->sizes, count($aData[0]['input']));
    if (!$this->hiddenSizes) {
      array_push($this->sizes, Math.max(3, Math.floor(count($aData[0]['input']) / 2)));
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
    
    $this->fnSetLogMethod($aOptions['log'] || $this->trainOpts['log']);
    $this->activation = $aOptions['activation'] || $this->activation;
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
  
  public function fnTrainingTick($aData, $aStatus, $iEndTime) 
  {
    if ($aStatus['iterations'] >= $this->trainOpts['iterations'] || $aStatus['error'] <= $this->trainOpts['errorThresh'] || time() >= $iEndTime) {
      return false;
    }

    $aStatus['iterations']++;

    if ($this->trainOpts['log'] && ($aStatus['iterations'] % $this->trainOpts['logPeriod'] === 0)) {
      $aStatus['error'] = $this->fnCalculateTrainingError($aData);
      $this->trainOpts.log("iterations: {$aStatus['iterations']}, training error: {$aStatus['error']}");
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
    return status;
  }
  
  /*
  trainAsync(data, options = {}) {
    let status, endTime;
    ({ data, status, endTime } = $this->_prepTraining(data, options));

    return new Promise((resolve, reject) => {
      try {
        const thawedTrain = new Thaw(new Array($this->trainOpts.iterations), {
          delay: true,
          each: () => $this->_trainingTick(data, status, endTime) || thawedTrain.stop(),
          done: () => resolve(status)
        });
        thawedTrain.tick();
      } catch (trainError) {
        reject({trainError, status});
      }
    });
  }
   */
  
  public function fnTrainPattern($aInput, $aTarget, $blogErrorRate) 
  {
    // forward propagate
    $this->fnRunInput($aInput);

    // back propagate
    $this->fnCalculateDeltas($aTarget);
    $this->fnAdjustWeights();

    if ($blogErrorRate) {
      return mse($this->errors[$this->outputLayer]);
    } else {
      return null;
    }
  }
  
  public function fnCalculateDeltasSigmoid($aTarget) 
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
  }
  
  public function fnCalculateDeltasRelu($aTarget) 
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
  }
  
  public function fnCalculateDeltasLeakyRelu($aTarget) 
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
  }

  public function fnCalculateDeltasTanh($aTarget) 
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
        $this->biases[$iLayer][$iNode] += $this->trainOpts['learningRate'] * delta;
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
    if (!is_array($mDatum)) {
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
        }
      );
    }

    if (!is_array($aData[0]['output'])) {
      if (!$this->outputLookup) {
        $this->outputLookup = Lookup::fnBuildLookup(
          array_map(function($v) { return $v['output']; })
        );
      }
      $aData = array_map(
        function($v) 
        {
          $aArray = Lookup::fnToArray($this->outputLookup, $v['output']);
          return array_merge($v, [ 'output' => $aArray ]);
        }
      );
    }
    
    return $aData;
  }
}
