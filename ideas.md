# model with invisible 'thinking' tokens trained via rl, with rewards based on supervised token accuracy
- [thinktoks](https://github.com/ekhadley/thinktokens) was a project I started but never saw results on due to training code being very complicated.
- during ordinary pretraining, the model autoregressively outputs tokens on which we calculate crossentropy against the correct token for each sequence position.
- the idea was to add a bunch of extra tokens to the models vocab.
- these tokens would be *ignored* when they are output during training. We continue spitting out tokens until we have produced as many 'real' tokens as were in the output sequence, ignoring all the extra 'thinking' tokens.
- During training, the model will receive no supervised feedback whatsoever when a thinking token is output.
- Instead, we include another loss term to the total loss for a sequence. The loss of a particular invisible token is a function of the ordinary, supervised loss for all the ordinary tokens which were output later.
- So the reward signal for normal tokens is simply to correctly predict the next token, like a normal model, except there are invisible tokens in the sequence which weren't part of the input sequence.
- The reward signal for the invisible tokens is the loss of all following real tokens.
- So we encourage the model to output normal tokens which are the same as the actual next token, and to output invisible tokens which increase the chance of outputting the correct prediction for the next token.
- It would also be probably necessary to apply a fixed cost for outputting invisible tokens to get it to eventually output real token predictions instead of 'thinking forever.'
- The idea is to allow the model to think, but not out loud. To be able to output intermediate results of unconstrained semantic meaning, and to do this for as many tokens as required.
- Language models contain in their weights algorithms whcih solve all/some of the problems whcih it is presented in the task of next token text prediction (IOI, arithmetic, etc). These algorithms are however limited in number of steps by the number of layers in the model.
- The idea of tihnking tokens is to allow them to discover and learn arbitrary length algorithms to apply to the problems present in NTP.
- The way chat models work is that they are originally trained to predict, then we set up the prompt so that their job is to predict the outputs of a character who happens to be a helpful AI assistantin conversation with a curious user. This can be done with just prompting but improved via fine tuning. The think out loud strategy allows the model to imitate the outputs of a human reasoning out loud, but there is evidence that the model is doing things somewhat different from what it appears. (see research on perturbing chains of thought and observing curiously unaffected final answers)
- Doing rl on chain of thought is hard (but is what openai's o-models are doing and its clearly working), but doing rl on next-token prediction is easy.
- thought: it's easier to do alternating next token prediction, and think-token rl training rather than both simultaneously. 
- The current generation of reasoning models are doing reasoning using invisible tokens trained via rl. The difference is that they are trained to reason in post treaining, and using normal tokens which are just not checked. This makes the largest issue finding supervision for the rl, which is why the o-models excel in math and code, where answers are verifiable and synthesizable.
- This would allow models to learn to reason during pretraining and all other places, becuase the rl reward is just next token prediction accuracy. This method can be applied on top of the other reasoning tech too.

# circuit mining via unsupervised component importance clustering
- take some set of inputs and model predictions.
- Create some set of metrics to establish which components of the model are important for a particular token prediction.
- Do unsupervised clustering to find sets of sequences which use a similair subset of model components to make
a prediction.
- Then zoom into some particular cluster and repeat, finding clusters of tightly related components within that data subset.
- This would allow us to zoom in to particular component groups which specialize for performing certain tasks.
- Assuming we are able to find nice identifiable clusters, this would show something akin to 'feature splitting' from
the SAE literature, where we can zoom into activations of a particular feature and find sub-features.
- Example of an interesting thing we might find: the model uses some sparse subset of components for doing math. We
perform clustering on just math problems, and find one subset of the subset is responsible for doing subtraction, one
for converting between currencies of different countries, one for calculating percentages, etc.
- Could mine a model for all/a bunch of its circuits and pair them with strogn activating examples, like neuronpedia.
- Then if one wanted to find the circuit responsible for some task, they would just have to produce a dataset which
demonstrates the capability, and we could use the circuit-pedia to identify which cluster the dataset's examples
are closest to.

# There are many methods of circuit discovery now. What do we do with the circuits
- Has there been any work in actually using discovered circuits to 'fix' or contrain models?
- good circut finding works have shown examples of using the supposedly critical circuit to generate adversarial examples. In other words to find the bugs in the algorithm the transformer implements.
- Can we use mechanistic understanding of the circuits to fix the models in a general way?
- like ROME for any general circuit?
- sounds pretty hard when you put it like that.
- but something in the vein of 'lets use all these circuits to do something actually useful/make models safer permanently or during inference'

# Examine yes/no arithmetic/simple math problems. what does each layer contribute to the answer across different problems.
- example: "a 6 pack of coke costs $10, a 4 pack costs $3.5. is the 6 pack a better deal?"
- different components will have different contributions to the yes/no logits.
- Examine different layers/components, seeing what direction each layer contributes on the yes-no spectrum.
- Change 'coke' in the example to 'pepsi'. are the same layers doing the same thing?
- I had this idea a long time ago and there was a specific angle i found really interesting but i cant remember it now...

# training encoder-decoder transformers to decompile program binaries
- while attending to the whole input binary file, autoregressively output the input file.
- For training data, I imagine we could use packages available via package managers. These often work by downloading source code from github or similair and then compiling locally.
- This gives us convenient access to a huge database real world programs in both source code and compiled format.

# general vein: mechinterp on reasoning models
- for exmaple, agentic-type reasoning models have to soemtimes realize "this approach isnt working let me try something else". Can we discover the circuit that triggers this?
- A reasoning model also has to choose when to stop thinking and start 'speaking'. Can the responsible circuit be discovered? Can it be intervened to make a model think more/less than it normally would?

# general vein: doing automated circuit discovery during pretraining. what changes over time?

# general vein: doing circuit discovery on models before/after some fine tuning
- try to see if we can find ubiquitious differences, potentially relating the different circuits to the kind of fine tuning we did.
- mechinterp on anthropics 'model organisms of misalignment' models? i dont see papers having done this, idk why. is it a bad idea?