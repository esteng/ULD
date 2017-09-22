# poster presentation outline

## describe the problem
- unsupervised learning intro (Elias) 
- language acquisition -- linguistic problem being solved (Emily)  

## the model 

- Broad structure (Emily) 
	- Broadly, the model is composed of 3 components: one which identifies similar units of sound and outputs a list of phone-like units given an audio recording, one which groups these units into repeated words and phrases, and a noisy channel model, which mediates between the two and allows substitutions, inserts, and deletes in the phone sequence.

- why this model (Elias) 
	- train on less speech data
	- linguistically accurate/interesting
- setting the stage for generative models (Emily)
	- When we describe the model in more detail, we're going to describe the generative model. The way our training works is that we make the assumption that any audio we're given as input was generated using this model, and then perform inference to maximize the likelihood of generating that audio data.


## our contribution
- Variational Bayes (Elias)
	- intro to why VB  
		- inference problem 
		- faster 
	- what is VB
		- high-level explanation of what you're optimizing    
	-why now?
		- VB implementations available 
		- more computing power available easily
- Noisy channel (Emily) 

## Model structure
- from the generative perspective, with relevant latent variables
- Adaptor grammar (Elias)
- Noisy channel
	- generative (Emily) 
	- inference (Elias) 
- DPHMM (Emily) 

## Future work
- Experiments to run (Emily) 
- possible engineering applications (Elias) 
	 