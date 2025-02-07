# try to make 'thinktoks' work.
- [thinktoks](https://githubcom/ekhadley/thinktokens) was a project I started but never saw results on due to training code being very complicated.
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
- I think it's plausible that this, or something like it, is actually what openai's o-models are doing behind the scenes. Having 'thinking' tokens whcih dont meaningfully translate to textual tokens could be why the o models output summaries of their thought process instead of the original tihng. but it could also be to hide whatever thinking strategies/thought processes they have trained the model to use.
- There are endless variations to this theme. You could have a multihead model, where one thinks and one speaks, or you could expand the vocab of a pretrained model and fine tune it to use the invisible tokens, etc. (I would guess openai does this last one since pretraining via autoregression is very not parallelizable)

# There are many methods of circuit discovery now. What do we do with the circuits
- Has there been any work in actually using discovered circuits to 'fix' or contrain models?
- good circut finding works have shown examples of using the supposedly critical circuit to generate adversarial examples. In other words to find the bugs in the algorithm the transformer implements.
- Can we use mechanistic understanding of the circuits to fix the models in a general way?
- like ROME for any general circuit?
- sounds pretty hard when you put it like that.
- but something in the vein of 'lets use all these circuits to do something actually useful/make models safer permanently or during inference'

# Examine yes/no arithmetic/simple math problems.
- example: "a 6 pack of coke costs $10, a 4 pack costs $3.5. is the 6 pack a better deal?"
- different components will have different contributions to the yes/no logits.
- Examine different layers/components, seeing what direction each layer contributes on the yes-no spectrum.
- Change 'coke' in the example to 'pepsi'. are the same layers doing the same thing?
- I had this idea a long time ago and there was a specific angle i found really interesting but i cant remember it now...

# general vein: doing automated circuit discovery during pretraining

# general vein: doing circuit discovery on models before/after some fine tuning
- try to see if we can find ubiquitious differences, potentially relating the different circuits to the kind of fine tuning we did.
