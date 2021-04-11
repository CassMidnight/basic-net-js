fs = require('fs');
mlp = require('./mlp.js');

let inputNodeCount = 4;
let outputNodeCount = 3;

let inputNodes = [];

for (i = 0; i < inputNodeCount; i++){
  inputNodes.push(mlp.createTemplateNode());
}

let layer1 = [];

for (i = 0; i < 10; i++){
  layer1.push(mlp.createTemplateNode());
}

let layer2 = [];

for (i = 0; i < 10; i++){
  layer2.push(mlp.createTemplateNode());
}

let outputNodes = [];

for (i = 0; i < outputNodeCount; i++){
  outputNodes.push(mlp.createTemplateNode());
}

let model = [inputNodes, layer1, layer2, outputNodes];

let irisData = [];

fs.readFile('irisTemp.csv', 'utf8', function (err,data) {
  let lines = data.split('\r\n');
  irisData = lines.map(line => {
    //0.5,0.33,0.14,0.02,1 
    let values = line.split(',');

    //console.log(values);

    let inputValues = values.slice(0, 4).map(val => parseFloat(val) ); 

    let outputValues = [];
    
    switch (values[4]){
      case '1': 
        outputValues = [1, 0, 0];
        break;
      case '2': 
        outputValues = [0, 1, 0];
        break;
      case '3': 
        outputValues = [0, 0, 1];
        break;
    } 

    return [ ...inputValues, ...outputValues ];
  });

  //console.log("irisData");
  //console.log(irisData);

  mlp.randomiseNodes(model);
  //mlp.randomiseNodes(model, true);

  //train
  train(model, irisData);
});

function train(model, data, inputValueCount, outputValueCount){

  let thirds = Math.trunc(data.length / 3);

  let trainingData = [];
  let testingData = [];
  let validationData = [];

  data.forEach(exPair => {
    switch(random(1,4)){
      case 1: 
        trainingData.push(exPair)
        break;
      case 2: 
        testingData.push(exPair)
        break;
      case 3: 
        validationData.push(exPair)
        break;
    }
  });

  //start with eta of 0.1
  let eta = 0.1;
  let presentations = 5;

  for (let k = 1; k <= presentations; k++){
    let sumPresentationError = 0;
    trainingData.forEach(exPair => {
      //console.log(exPair);

      let forwardOutput = mlp.forwardPass(model, exPair);
      //console.log(forwardOutput);

      let totalError = mlp.backwardPass(model, exPair, eta);
      //console.log(totalError);
      
      sumPresentationError += totalError;
    })
    console.log(k + "/" + presentations, sumPresentationError/trainingData.length);
  }

  console.log(model[model.length - 1].forEach(node => {
    console.log(node.value)
  }))

  mlp.niceLogModel(model)

  const testingDataResult = mlp.evaluateOnList(model, testingData);

  console.log(testingDataResult);

  const validationDataResult = mlp.evaluateOnList(model, validationData);

  console.log(validationDataResult);
}

function random(min = 0, max = 100) {
  let num = Math.random() * (max - min) + min;
  return Math.floor(num);
};