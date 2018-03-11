const synaptic = require('synaptic');
const R = require('ramda')
var fs = require('fs');
const util = require('./util.js');


var contents = fs.readFileSync('./data/makes.txt', 'utf8');
var makes = contents.toLowerCase().split(',');
// makes = ["toyota"];

contents = fs.readFileSync('./data/make-models.txt', 'utf8');
contents = contents.toLowerCase();

const dictionary = util.createSlugDictionary();
console.log("dictionary: ", dictionary)

const OUTPUT_DIMENSION = 1;
const INPUT_DIMENSION = Object.keys(dictionary).length;
const HIDDEN_SIZE = 20;

var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

// 26 characters + 10 digits +
var LSTM = new Architect.LSTM(INPUT_DIMENSION, HIDDEN_SIZE, OUTPUT_DIMENSION);

var contentsTruth = R.map(c => c == '_' ? 1 : 0)(util.replaceKeyWordsWithSymbol(contents, makes, '_'))

// console.log(util.displayResultsInCap(contents, contentsTruth))

console.log("Training...")

const TRANING_ROUNDS = 70;

const MIN_LEARNING_RATE = 0.001;
var learningRate = 0.5;
var lastS = undefined;
var initGain = undefined;

R.times(R.identity, TRANING_ROUNDS).forEach(()=> {

	var s = R.compose((s) => {
		// adjust learning rate
		if(learningRate > MIN_LEARNING_RATE) {
			var gain = lastS - s;
			console.log("gain: ", gain, " initGain: ", initGain)
			if(lastS) {
				gain = lastS - s;
				if(!initGain) {
					initGain = gain;
				} else {
					if(gain / initGain < 0.5) {
						learningRate /= 2;
						console.log("set learningRate: ", learningRate);
						initGain = gain;
					}
				}
			}
			lastS = s;
		}
		return s;
	}, R.sum, R.map(pair => {
		const char = pair[0];
		const expectedOutput = pair[1];

		const input = util.charToVector(char, dictionary);
		const expectedOutputVector = util.intToVector(expectedOutput, OUTPUT_DIMENSION)

		var output = LSTM.activate(input)
		LSTM.propagate(learningRate, expectedOutputVector)

		const dist = util.distance(output, expectedOutputVector);
		// console.log("output: ", output, " expected: ", expectedOutputVector);
		// console.log("distance: ", dist);
		return dist

	}))(R.zip(contents, contentsTruth))

	console.log("dist: ", s)
})


console.log("Classifying...")


function classifyText(contents) {
	contents = contents.toLowerCase();
	return R.map(ch => {
		const input = util.charToVector(ch, dictionary);
		const r = LSTM.activate(input);
		// console.log(r);
		return r;
	})(contents)
}

// contents = fs.readFileSync('./data/make-model-trims.txt', 'utf8');
// contents = contents.toLowerCase();
//
// contents = contents.substring(0,1000)

contents = "do you believe that toyota is totally affordable. hoda accord is the most common model in america market.";
var results = classifyText(contents)
console.log("results: ", results);
console.log( output = util.displayResultsInCap(contents, results, 0.5));



// fs.writeFile("/tmp/result.txt", output);
