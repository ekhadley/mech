# model with invisible 'thinking' tokens trained via rl, with rewards based on supervised token accuracy
- [thinktoks](https://github.com/ekhadley/thinktokens) was a project I started but never saw results on due to training code being very complicated.
- during ordinary pretraining , the model autoregressively outputs tokens on which we calculate crossentropy against (normal supervised loss) the correct token for each sequence position.
- the idea was to add a bunch of extra tokens to the models vocab.
- these tokens would be *ignored* when they are output during training (when calculating the loss that is). We continue spitting out tokens autoregressively (like during model inference) until we have produced as many 'real' tokens as were in the output sequence, ignoring all the extra 'thinking' tokens.
- During training, the model will receive no supervised feedback whatsoever when a thinking token is output.
- Instead, we include another loss term to the total loss for a sequenceed   The loss of a particular invisible token is a function of the ordinary, supervised loss for all the ordinary tokens which were output later.
- So the reward signal for normal tokens is simply to correctly predict the next token, like a normal model, except there are invisible tokens in the context which weren't part of the input sequence, but can be attended to for next token prediction.
- The reward signal for the invisible tokens is a function of the loss of all following real tokens.
- So we encourage the model to output normal tokens which are the same as the actual next token, and to output invisible tokens which increase the chance of outputting the correct prediction for the next token.
- It would also be probably necessary to apply a fixed cost for outputting invisible tokens to get it to eventually output real token predictions instead of 'thinking forever.'
- The idea is to allow the model to think, but not out loud. To be able to output intermediate results of unconstrained semantic meaning, and to do this for as many tokens as required.
- Language models contain in their weights algorithms whcih solve the problems which it is presented with in the task of next token text prediction (IOI, arithmetic, etc). These algorithms are however limited in number of steps by the number of layers in the model. (https://arxiv.org/abs/2210.10749)
- The idea of hidden tokens is to allow them to discover and learn arbitrary length (in time or space) algorithms to apply to the problems present in NTP.
- The current generation of reasoning models are doing reasoning using invisible tokens trained via rl. The difference is that they are trained to reason in post training, and using normal tokens which are just not checked. This makes the largest issue finding supervision for the rl, which is why the recent reasoning models are mainly improved in just math and code, where answers are verifiable and questions are synthesizable.
- This would allow models to learn to reason during pretraining and all other places, becuase the rl reward is just next token prediction accuracy. This method can be applied on top of the other reasoning tech too.
- I have reently found out about the COCONUT paper from december of 2024. It describes something similair to this. They also use augmented tokens used for thinking, which are ignored when calculting supervised loss. They differ in that they do not expand the dictionary or unembed these special tokens, and simply leave thought tokens as continuous when they start the next token generation. They also use <thinking> tags to enclose a thinking segment of a set number of augmented tokens.
    - They make no mention of rl. The paper made me realized that the rl is not necessary at all. The gradient for th esupervised loss will propogate through the thought tokens naturally, as the normal supervised token positions will attend to and read info from the thought token. so yeah.
    - The main difference here is that they are still only doing reasoning training as part of post training, and still only using chain of though in post trainin. The method I propose is a deeper modification to the fundamental operation of the transformer, and involves reasoning which happens on the most fundamental level, including during pretraining, post training, and basically any time the model is outputting tokens.



# circuit mining via unsupervised component importance clustering
- take some set of inputs and model predictions.
- Create some set of metrics to establish which components of the model (all the MLP layers, all the attention heads, all the layer norms, etc) are important (contribute strongly to the output) for a particular token prediction.
- Empirically we often find that many layers of a deep, overparameterized neural network are unimportant for a particular inference.
- For each token prediction, we check each component to see how important it was.
- Do unsupervised clustering to find groups of model components which are tightly related. As in, find sets of components which are often all highly important at the same time, or all basically unimportant at the same time. These components are likely coordinating in some way to produce a specific model behavior.
- Then zoom into some particular cluster and repeat, finding clusters of tightly related components within that data subset.
- This would allow us to zoom in to particular component groups which specialize for performing certain tasks.
- Assuming we are able to find nice identifiable clusters, this would show something akin to 'feature splitting' from
the SAE literature, where we can zoom into activations of a particular feature and find sub-features.
- Example of an interesting thing we might find: the model uses some sparse subset of components for doing math. We
perform clustering on just math problems, and find one subset of the subset is responsible for doing subtraction, one
for converting between currencies of different countries, one for calculating percentages, etc.
- Once we have a map of the 'behavior clusters', we can go through the sample dataset, and based on which inputs are activating in which cluster, we can try to estimate what exactly the behavior is that the group of components are coordinating to perform.
- You could do this recursively, going all the way down, making some map of nested behaviors based on the relevant components for that behavior, sort of like neuronpedia and similar projects.
- Or going backwards: given a dataset requiring some capability, we could easily just run the model on it, find out which components are important, and see where that set of importances lie relative to the other clusters.

# There are many methods of circuit discovery now. What do we do with the circuits
- Has there been any work in actually using discovered circuits to 'fix' or contrain models?
- good circut finding works have shown examples of using the supposedly critical circuit to generate adversarial examples. In other words to find the bugs in the algorithm the transformer implements.
- Can we use mechanistic understanding of the circuits to fix the models in a general way?
- like ROME for any general circuit?
- sounds pretty hard when you put it like that.
- but something in the vein of 'lets use all these circuits to do something actually useful/make models safer permanently or during inference'

# Examine yes/no arithmetic/simple math problems. what does each layer contribute to the answer across different problems, and how does the contribution of each layer change as the numerical inputs to the math problem change?
- example: "a 6 pack of coke costs $10, a 4 pack costs $3.5. is the 6 pack a better deal?"
- different components will have different contributions to the yes/no logits.
- Examine different layers/components, seeing what direction each layer contributes on the yes-no spectrum.
- if we change the values in the problem, e.g. increase the cost of the cokes to 11 dollars, how does the yes-no contribution of each layer change?
- It seems like we could sort of take the gradient of each layer's yes-no contribution with respect to the inputs of the math problem, and use that to figure out specifically which operation each layer performs.
- This is only possible because of the yes/no binary answer, meaning the value of interest for each component is just a scalar, as opposed to the usual mechinterp challenge of trying to decipher what is the meaning of the gigantic vector that layer xyz just added into the residual stream and got read in by layer abc ...

# rl on chain of thought in game playing domains
- current llm reasoning-through-rl approaches require verifiable domains.
- therefore the main domains, where improvements have been the largest for reasoning models, are math and code.
- games are also a verifiable domain. (the one who won probably played better)
- could you do rl on train of thought in an adversarial game setting?
- With an llm you could even have the training include multiple different games. go, chess, etc all at the same time.
- If it works, this could potentially open the door to do reasoning-rl on more domains. Domains in which solutions are not objectively verifiable, but where you can objectively say which of two solutions is better.
- Risk (Mafia, etc) involve conversation, negotiation, or deception as central mechanics. How well can models learn to outwit other models in conversation?
    - lol, maybe they discover and deploy jailbreak sequences to cause other models to go haywire

# transformer models for program binary decompilation
- while attending to the whole input binary file, autoregressively output the source code.
- For training data, I imagine we could use packages available via package managers (pip, pacman, apt, flatpak). These often work by downloading source code from github or similair and then compiling locally.
- This gives us convenient access to a huge database real world programs in both source code and compiled format.
- probably would want to output a serialized AST or something instead of normally tokenized text.
- But it would be cool to see if it can figure out likely variable names and stuff.

# transformer models for 3d object generation
 - fine tune an instruct models on 3d object files associated with object descriptions.
    - what kind of 3d object file?
        - most object files contain a set of unordered vertices
 - can we then get it to generate object files from user descriptions?
 - actually can u fine tune instruct models on new data like this and get it to follow instructions properly? i do not know

# general vein: mechinterp on reasoning models
- for exmaple, agentic-type reasoning models have to soemtimes realize "this approach isnt working let me try something else". Can we discover the circuit that triggers this? Is it absent in non-reasoning models?
- A reasoning model also has to choose when to stop thinking and start 'speaking'. Can the responsible circuit be discovered? Can it be intervened upon to make a model think more/less than it normally would?

# general vein: doing automated circuit discovery during pretraining. what changes over time?

# general vein: doing circuit discovery on models before/after some fine tuning
- try to see if we can find ubiquitious differences, potentially relating the different circuits to the kind of fine tuning we did.
- some circuits are probably being created during fine tuning. Do any go away, or are they just suppressed?
- mechinterp on anthropics 'model organisms of misalignment' models? i dont see anyone having done this, idk why it seems kind of obvious. is it a bad idea?

# how good are models at noticing confusion?
- an important part of the whole agentic language model, feedback from the environment thing is noticing when the outputs from the environment are unexpected or confusing.
- a very common failure mode of these models is when they don't pick up on when something is amiss, something that would challenge their assumptions and cause them to rethink their plans.
- Language models are trained as language predictors, so one might expect them to be very good at noticing unlikely tokens, but the only info a model really has available is in the CoT, which logit outputs are not.
- The project would be to test how good models are at identifying outputs which were unlikely, given the model's understanding of the environment generating the data.
- Could use simple programs whcih print to the terminal, showing the model the output, then simply asking it to say how unlikely it thinks that output was.
- We could then switch out the proper program output with an incorrect one, and see if the model is able to identify that the output is unlikely, or does it succumb to hindsight bias and say that everything it already knows was likely a priori.
- could test with models with different training recipes (one shot, reasoning models, etc)
