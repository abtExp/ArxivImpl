# Spatio-Temporal Video Autoencoder With Differentiable Memory
(arxiv:1511.06309)

```
@inproceedings{PatrauceanHC16,
  author    = {Viorica P{\u a}tr{\u a}ucean and
               Ankur Handa and
               Roberto Cipolla},
  title     = {Spatio-temporal video autoencoder with differentiable memory},
  booktitle = {International Conference on Learning Representations (ICLR) Workshop},
  year      = {2016}
}
```

Read The Complete paper <a href='https://arxiv.org/abs/1511.06309'>here</a><br />
Torch code from the authors available <a href='https://github.com/viorik/ConvLSTM'>here</a>

<br />

#### Summary

This paper proposes a new architecture for a video autoencoder by using a spatial encoder and a nested temporal encoder.

The model works as follows :
1. At time step 't', input the video frame <b>f\<t\></b> .<br /><br />

2. The *<b>spatial encoder</b>* (ConvNet) produces an encoding <b>e\<t\></b>.<br /><br />.

3. The *<b>temporal encoder</b>* receives this encoding after being projected in temporal space and generates the hidden temporal activation <b>h\<t\></b> by also integrating the temporal info stored from the previous frames.

######  The LSTM uses <a href='https://arxiv.org/abs/1506.04214'>convLSTM</a> architecure,so it uses convolutions instead of linear operations, thus greatly reducing the parameters as well as producing encoding in the same layout as the spatial encoder.


<b><i>i<sub>t</sub></i></b> = σ(x<sub>t</sub> ∗ w<sub>xi</sub> + h<sub>t-1</sub> ∗ w<sub>hi</sub> + w<sub>ibias</sub>)<br />
<b><i>f<sub>t</sub></i></b> = σ(x<sub>t</sub> ∗ w<sub>xf</sub> + h<sub>t−1</sub> ∗ w<sub>hf</sub> + w<sub>fbias</sub>)<br />
<b><i>c˜<sub>t</sub></i></b> = tanh(x<sub>t</sub> ∗ w<sub>xc˜</sub> + h<sub>t−1</sub> ∗ w<sub>hc˜</sub> + w<sub>c˜bias</sub>)<br />
<b><i>c<sub>t</sub></i></b> = c˜<sub>t</sub> <b>.</b> i<sub>t</sub> + c<sub>t−1</sub> <b>.</b> f<sub>t</sub><br />
<b><i>o<sub>t</sub></i></b> = σ(x<sub>t</sub> ∗ w<sub>xo</sub> + h<sub>t−1</sub> ∗ w<sub>ho</sub> + w<sub>obias</sub>)<br/>
<b><i>h<sub>t</sub></i></b> = o<sub>t</sub> <b>.</b> tanh(c<sub>t</sub>)<br />




4. The <b>h\<t\></b> is fed into an *<b>optical flow module</b>* that generates a dense transformation map giving the sense of the flow or movement of objects in the frame in the next time frame.
For This 2 convolutions with (15x15) kernels are used.

To Ensure smoothness in the motion and avoid randomness in between frames,local gradient of the flow map <b>∇T</b> is penalized using *<b>Huber Loss Smoothness penalty</b>*.
It is used as it has edge preserving capabilities.

<b>H<sub>δ</sub>(a<sub>ij</sub>)</b> = <b>{</b>
 (1/2)a<sub>ij</sub><sup>2</sup>  ,   for <b>|</b>a<sub>ij</sub><b>|</b> ≤ δ
δ(<b>|</b>a<sub>ij</sub><b>|</b> − (1/2)δ)     ,otherwise
										<b>}</b>

<b>∇H<sub>δ</sub>(a<sub>ij</sub>)</b> = <b>{</b>
a<sub>ij</sub> ,   for |a<sub>ij</sub>| ≤ δ,
δsign(a<sub>ij</sub>), otherwise
<b>}</b>


###### δ = 10<sup>-3</sup>, a<sub>ij</sub> represents elements of <b>∇T</b>


5. This smoothened optical flow is passed through a *<b>grid generator</b>* which holds the initial position for each transformed pixel.

6. The *<b>Sampler</b>* receives <b>e<sub>t</sub></b> and the grid from the grid generator and produces the prediction for the next frame (<b>y<sup>^</sup><sub>t</sub></b> = prediction for <b>x<sub>t+1</sub></b>)

 Given the flow map <b>T</b> , Grid Generator computes for
each element in the grid the source position (x<sub>s</sub>, y<sub>s</sub>) in the input feature map from where S needs to sample to fill the position (x<sub>o</sub>, y<sub>o</sub>)


7. The *<b>Spatial Decoder</b>* Produces the prediction for the next frame having dimensions same as the input.

8. The *<b>Loss Function</b>* is the reconstruction error between the predicted and the ground truth next frame.

<b><i>L<sub>t</sub></i></b> = ||<b>Y˜<sub>t+1</sub></b> − <b>Y<sub>t+1</sub></b>||<sup>2</sup><sub>2</sub>+ <b>w<sub>H</sub>H</b>(grad(T))

<b>w<sub>H</sub></b> is hardcoded as <b>w<sub>H</sub></b> = 10<sup>-2</sup>.

_____________________________________________________________________________________

#### Architecture

1. *<b>The Spatial Encoder and Decoder (ConvNet)</b>*
Consists of a single layer with 16 (7x7) filters.
<br/>

2. *<b>The LSTM</b>*
Has 64 (7x7) filters
<br/>

3. *<b>The Optical Flow Regressor</b>*
Has 2 conv layers each with 2 (15x15) filters and a (1x1) conv layer.

*<b>The Grid Generator, Huber Loss and the Sampler have no trainable parameters.</b>*

<img src='./spation.png' alt='spatio-temporal video autoencoder'/>


###### Training was done using RMSProp with lr = 10<sup>-4</sup> and decay of 0.9 after every 5 epochs. Spatial Encoder was initialized using <u>xavier</u> initializer. The LSTM weights were initialized from a unform distribution U(-0.08,0.08). Biases except the forget gate were 0 and at forget gate were 1. Gradient Clipping was used.


_____________________________________________________________________________________
