const synaptic = require('synaptic');

var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

var LSTM = new Architect.LSTM(6,7,2);

var myTrainer = new Trainer(LSTM);
myTrainer.DSR({
    targets: [2,4],
    distractors: [3,5],
    prompts: [0,1],
    length: 10
})

LSTM.activate([1,2,3,4,5,6])

LSTM.propagate(0.01, [0,1])
