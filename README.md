# Unsupervised Lexicon Discovery (Symbol-to-audio mapping component)

The Unsupervised Lexicon Discovery model identifies a lexicon of phrase, word, and sub-word units in unlabeled speech audio input.

This is a Variational Bayesian implementation of the phonetic-phonological component of the model, which learns a sequence of noisy-channel edit operations to transform a given high-level symbol sequence into a lower-level representation, and a mapping from this lower-level sequence onto the corresponding audio.

The implementation is an extension of the Acoustic Model Discovery Tool Kit by Lucas Ondel et al., located at https://github.com/amdtkdev/amdtk.
