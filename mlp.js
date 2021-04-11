let nodeTemplate = {
  // Feed Forward
  netValue: 0.5,
  value: 0.5,
  bias: 0,
  weights: [],

  // Back Propagation
  error: 0,
  derivative: 0,
};

function createTemplateNode(){
  let newTemplate = {...nodeTemplate};
  Object.keys(newTemplate).forEach(prop => {
    if ( Array.isArray( newTemplate[prop] ) ){
      newTemplate[prop] = [];
    } else if ( typeof newTemplate[prop] === 'object') {
      throw "Error! Function createTemplateNode does not support deep copy!"
    }
  })
  return {...nodeTemplate, weights: [], newWeights: []}
}

function randomiseNodes(model, debugMode = false){
  for(let layerIndex = 0; layerIndex < model.length; layerIndex++){
    for(let nodeIndex = 0; nodeIndex < model[layerIndex].length; nodeIndex++){

      let inputLayerIndex = layerIndex - 1;

      //skip the input layer since it has no weights
      if (inputLayerIndex < 0){
        continue; 
      }
      //console.log(inputLayerIndex);
      let inputLayerLength = model[inputLayerIndex].length;

      //console.log('randomiseNodes', inputLayerIndex, inputLayerLength)

      for(let j = 0; j < inputLayerLength; j++ ){
        if (debugMode){
          // Make the generation predictable for testing
          model[layerIndex][nodeIndex].weights.push(0.1 * (nodeIndex + layerIndex + j));
        } else {
          model[layerIndex][nodeIndex].weights.push(Math.random());
        }
        
      }

      //console.log(model[layerIndex][nodeIndex].weights);

      // You may want to comment this back in later
      //model[layerIndex][nodeIndex].bias = Math.random();
    }
  }
  //logObject("randomised model", model);
}

function evaluateOnList(model, exPairList, presentations){
  let result = {
    correct: 0,
    incorrect: 0,
    accuracy: 0.0,
    presentations: []
  };

  const inputCount = model[0].length;
  const outputCount = model[model.length - 1].length;

  exPairList.forEach(exPair => {
    const actualResult = forwardPass(model, exPair);
    const expectedResult = [...exPair].slice(inputCount, inputCount + outputCount);
    result.presentations.push({expected: expectedResult, actual: actualResult});
  });

  result.presentations.forEach(presentation => {
    let topIndex = -1;

    presentation.actual.forEach((number, index) => {
      if (topIndex === -1 || number > presentation.actual[topIndex]){
        topIndex = index;
      }
    });
    
    if(presentation.expected[topIndex] === 1){
      result.correct++;
    }
  });  

  result.incorrect = exPairList.length - result.correct;
  result.accuracy = (result.correct/exPairList.length) * 100;

  return result;
}

function forwardPass(model, inputData){
  inputData.forEach((dataPoint, index) => {
    if (index < model[0].length){
      model[0][index].value = dataPoint;
    }
  })

  for (let i = 0; i < model.length - 1; i++){
    calculateLayers(model[i], model[i+1])
  }

  //Return an array of the values of the nodes
  return model[model.length -1].map(node => {
    return node.value;
  })
}

function calculateLayers(layer1, layer2){
  for(nodeIndex = 0; nodeIndex < layer2.length; nodeIndex++){
    let node = layer2[nodeIndex];
    let sumOfWeights = node.weights.reduce((acc, weight, index) => {
      return acc + (weight * layer1[index].value);
    }, 0);
    
    node.netValue = sumOfWeights + node.bias;
    node.value = sigmoid(sumOfWeights + node.bias);
  };
}

function backwardPass(model, exPair, eta){
  let totalError = calculateError(model, exPair);

  // ----- Output Layer -----
  const outputLayerIndex = model.length - 1;
  for (let outputNodeIndex = 0; outputNodeIndex < model[outputLayerIndex].length; outputNodeIndex++){
    let node = model[outputLayerIndex][outputNodeIndex];
    let partialDerivativeOfNode = slope(sigmoid, node.netValue) * node.error;
    node.derivative = partialDerivativeOfNode;
    
    for (let weightIndex = 0; weightIndex < node.weights.length; weightIndex++){
      let weightDelta = eta * model[outputLayerIndex - 1][weightIndex].value * partialDerivativeOfNode;
      node.weights[weightIndex] += weightDelta;
    }
  }

  // ----- Hidden Layers ----- //
  // layerIndex > 0 ignores the first layers since we don't care about the input layer
  for (let layerIndex = outputLayerIndex - 1; layerIndex > 0; layerIndex--){
    for (let nodeIndex = 0; nodeIndex < model[layerIndex].length; nodeIndex++){
      let node = model[layerIndex][nodeIndex];

      for (let weightIndex = 0; weightIndex < node.weights.length; weightIndex++){
        //console.log(layerIndex, nodeIndex, weightIndex)
    
        let sumOfForwardLayersNodesPartialDerivativesAndWeights = model[layerIndex + 1].reduce((acc, forwardNode) => {
          //console.log(forwardNode.derivative, forwardNode.weights[nodeIndex]);
          return acc + (forwardNode.derivative * forwardNode.weights[nodeIndex]);
        },0);

        let partialDerivativeOfNode = slope(sigmoid, node.netValue) * sumOfForwardLayersNodesPartialDerivativesAndWeights;
        node.derivative = partialDerivativeOfNode;

        let weightDelta = eta * model[layerIndex - 1][weightIndex].value * partialDerivativeOfNode
        node.weights[weightIndex] += weightDelta;
      }
    }
  }

  return totalError;
}

// Based vaguely on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
function calculateError(model, exPair){
  const inputCount = model[0].length;
  const outputCount = model[model.length - 1].length;

  let expectedOutput = exPair.slice(inputCount, inputCount + outputCount);
  let layerIndex = model.length - 1;

  let totalError = 0;
  let outputLayer = model[layerIndex];
  for (let nodeIndex = 0; nodeIndex < outputLayer.length; nodeIndex++){

    outputLayer[nodeIndex].error = expectedOutput[nodeIndex] - outputLayer[nodeIndex].value;

    // We need to total error of the layer for later
    totalError += Math.pow(outputLayer[nodeIndex].error, 2);
  }

  return totalError;
}

function sigmoid(x) {
  return 1 / ( 1 + Math.pow( Math.E, -1 * x) );
}

function slope (f, x, dx) {
  dx = dx || 0.00000000000001;
  return (f(x+dx) - f(x)) / dx;
}

/*
// From: https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
//0.21492890298354272
function slope (f, x) {
  return f(x) * (1 - f(x));
}
*/

function getOutputValues(model){
  return model[model.length - 1].reduce((acc, node) => {
    return [...acc, node.value];
  }, []);
}

function niceLogModel(model){
  console.group('model');
  model.forEach((layer, index) => {
    console.group(`layer ${index}`);
    console.log(index, JSON.stringify(layer, null, 2))
    console.group(`layer ${index}`);
  })
  console.groupEnd('model');
}

module.exports = {
  createTemplateNode,
  randomiseNodes,
  forwardPass,
  backwardPass,
  evaluateOnList,
  getOutputValues,
  niceLogModel
}