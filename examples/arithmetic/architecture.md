# Demonstration architecture

Byte-level recurrent network: 259-token vocabulary (UTF-8 bytes plus BOS, separator
and EOS), 24 hidden units, one tanh recurrent layer and linear output projection.
Train all parameters by full BPTT and SGD with answer-only cross entropy. Decode
greedily. C uses the recurrent hidden states of question bytes alone.
