# EncoderLSTMHybridModel
Encoder based transformer and LSTM hybrid model
This is a variable Encoder and LSTM depth Hybrid Model made for dialect detection in auditory datasets

This model implements an encoder only transformer architecture with multiheaded attention along with a basic LSTM model
The depth of both the encoder and LSTM are arbitrary along with the number of heads in the multiheaded attention.
This is done so help tune the performance to whatever hardware that is being used, since an increase in those 3 parameters greatly increases the amount of VRAM used while conversely greatly increasing the models accuracy.

The data being passed in, is a variable length speech .wav file which is being processed by taking the FFT of it using the pytorch vision spectrogram function and also adding on positional encoding based on the positinal encoding give in the attention is all you need paper.

The highest accuracy given from this model is 21% on dev, with 8 LSTM layers, 8 attention heads, and 32 encoder depth with 4gb of VRAM and 4 cores.

Given recent studies which show greatly increased performance of transformers on larger systems this accuracy is most likely not the maximum this model can produce. Such will be studied and explored further later on when such opportunities arrise.
