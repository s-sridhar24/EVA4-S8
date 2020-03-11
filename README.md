# EVA4-S8

FIles for this assignment are in this repo

Interesting results:

Batch of 4 performed better (84%) at 15 epochs, just increasing batch size to 16, 32, 64 and 128 dropped performance
transforms of images also reduced performance for same epochs
Tried switching to ReLU before addition, used nll and log_softmax. All these reduced performance though seem like logical.

Finally, this was a useful trick:
run 15 epochs, then run the same cell again. That helps break out of local minima rut it was stuck at.


