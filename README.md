# Energy-Based Models for Cross-Modal Localization using Convolutional Transformers

If you find this repository useful for your research, please cite our [paper](ECML_ICRA.pdf):

        @inproceedings{wu2022ecml,
              title={Energy-Based Models for Cross-Modal Localization using Convolutional Transformers},
              author={Alan Wu and Michael S. Ryoo},
              year={2022}
        }
        
We present a novel framework using Energy-Based Models (EBMs) for localizing a ground vehicle mounted with a range sensor against satellite imagery in the absence of GPS. Lidar sensors have become ubiquitous on autonomous vehicles for describing its surrounding environment. Map priors are typically built using the same sensor modality for localization purposes. However, these map building endeavors using range sensors are often expensive and time-consuming. Alternatively, we leverage the use of satellite images as map priors, which are widely available, easily accessible, and provide comprehensive coverage. We propose a method using convolutional transformers that performs accurate metric-level localization in a cross-modal manner, which is challenging due to the drastic difference in appearance between the sparse range sensor readings and the rich satellite imagery. We train our model end-to-end and demonstrate our approach achieving higher accuracy than the state-of-the-art on KITTI, Pandaset, and a custom dataset.

Below is the system diagram of our approach. We use lidar data to find vehicle pose within a satellite image. Point cloud data from range sensors have shown to be
effective in localization tasks when a flattening step is applied to produce a birds-eye view (BEV) of the environment surrounding the vehicle. We use this BEV representation to localize within a large map area. While our primary objective is to estimate vehicle position in the x,y-plane, our method also solves for the rotational offset between the online lidar image and the satellite map prior.
<\br>

![SystemDiagram](/figures/SystemDiagram_smallest.png)
<\br>

The key component of our EBM is a convolutional transformer (CT). The CT architecture applies particularly well to our application as the early convolutional
layers help better preserve local structural information that would otherwise have been lost by direct tokenization of input images (i.e. patch boundaries). The class and position embeddings used in the ViT architecture are eliminated in the CT approach. The implementation of the sequence pooling layer renders such embeddings unnecessary, leading to a more compact transformer. Our model architecture is shown below.

![SystemArch](/figures/CLECT.png)

We train the convolutional transformer (CT) in a self-supervised manner to construct the predicted satellite tile ˆIS from a weighted ensemble of tile candidates. It takes as input a lidar-satellite pair (ILSi = IL ⊕ISi ) concatenated along the channel axis where ILSi ∈ RH×W ×C . We use two convolutional layers with ReLU activation and max pooling to obtain an intermediate representation z ∈ RH0×W0×p so that p is equivalent to the embedding dimension of the transformer backbone used in
ViT. We reshape z to z0 ∈ Rl×p where l = H0 × W0, which is the length of the embedding sequence. z0 is then fed to the transformer encoder. The sequential output of
the M-layer transformer encoder is followed by a sequence pooling step [10] where the importance weights are assigned for the output sequence, which is subsequently mapped to a similarity score for the lidar-satellite pair in the MLP head. Each batch consists of physically proximal and distant lidar-satellite pairs with NS pairs in total, and thus yielding A = [α1, α2, ...αNS ]T ∈ RNS , which we use as similarity scores at inference.

We also built an EBM with a CNN backbone (i.e. ECML-CNN). It takes the same input as the EBM with a CT backbone (i.e. ECML-CT) and also outputs lidar-satellite similarity scores, as well as attempt to predict the satellite image reflecting the location of the vehicle. More details are shown in the Appendix of our paper. Below are predicted satellite images from both models compared to ground truth. For each set, we show CNN-predicted (left), CT-predicted (middle), and ground truth (right).

![SatPred](/figures/sat_pred.png)

We also compute the SSIM and PSNR to assess the quality of the generated images, which also reflects the accuracy of the localization prediction.  Higher is better.
![ssim_psnr](/figures/ssim_psnr.png)

Here are some examples of online vehicle lidar image queries overlaid on the bottom-left corner of satellite map priors and resulting localizations for KITTI (top), PandaSet (middle), and Custom (bottom) datasets. Black, red, and yellow boxes are ground truth, CT-predicted, and CNN-predicted locations, respectively. The maps are retrieved from a database so that the vehicle location is randomly positioned on the map to make the task more challenging and allow for one-shot localization.

![Results1](/figures/loc1.png)

Examples of ECML-CT successful predictions but ECML-CNN failures. Note the similarity in appearance between the true
locations and the incorrect predictions. Black, red, and yellow boxes are true, CT-predicted, and CNN-predicted locations, respectively.
![Results2](/figures/loc2.png)


![ResultsTable](/figures/results_table.png)
![hist_ct](/figures/histogram_ct.png)![hist_cnn](/figures/histogram_cnn.png)



