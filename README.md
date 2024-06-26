# Energy-Based Models for Cross-Modal Localization using Convolutional Transformers

If you find this repository useful for your research, please cite our [paper](ECML_ICRA.pdf):

        @inproceedings{wu2022ecml,
              title={Energy-Based Models for Cross-Modal Localization using Convolutional Transformers},
              author={Alan Wu and Michael S. Ryoo},
              year={2022}
        }
        
We present a novel framework using Energy-Based Models (EBMs) for localizing a ground vehicle mounted with a range sensor against satellite imagery in the absence of GPS. Lidar sensors have become ubiquitous on autonomous vehicles for describing its surrounding environment. Map priors are typically built using the same sensor modality for localization purposes. However, these map building endeavors using range sensors are often expensive and time-consuming. Alternatively, we leverage the use of satellite images as map priors, which are widely available, easily accessible, and provide comprehensive coverage. We propose a method using convolutional transformers that performs accurate metric-level localization in a cross-modal manner, which is challenging due to the drastic difference in appearance between the sparse range sensor readings and the rich satellite imagery. We train our model end-to-end and demonstrate our approach achieving higher accuracy than the state-of-the-art on KITTI, Pandaset, and a custom dataset.

Below is the system diagram of our approach. We use lidar data to find vehicle pose within a satellite image. Point cloud data from range sensors have shown to be
effective in localization tasks when a flattening step is applied to produce a birds-eye view (BEV) of the environment surrounding the vehicle. We use this BEV representation to localize within a large map area. While our primary objective is to estimate vehicle position in the x,y-plane, our method also solves for the rotational offset between the online lidar image and the satellite map prior.<br/><br/><br/>

![SystemDiagram](/figures/SystemDiagram_smallest.png)<br/><br/><br/>

The key component of our EBM is a convolutional transformer (CT). The CT architecture applies particularly well to our application as the early convolutional
layers help better preserve local structural information that would otherwise have been lost by direct tokenization of input images (i.e. patch boundaries). The class and position embeddings used in the ViT architecture are eliminated in the CT approach. The implementation of the sequence pooling layer renders such embeddings unnecessary, leading to a more compact transformer. Our model architecture is shown below.<br/><br/><br/>

![SystemArch](/figures/CLECT.png)<br/><br/><br/>

We train the convolutional transformer (CT) in a self-supervised manner to construct the predicted satellite tile &Icirc;<sub>S</sub> from a weighted ensemble of tile candidates. It takes as input a lidar-satellite pair (I<sub>LS<sub>i</sub></sub> = I<sub>L</sub> ⊕ I<sub>S<sub>i</sub></sub> ) concatenated along the channel axis where I<sub>LS<sub>i</sub></sub> ∈ &reals;<sup>H×W×C</sup> . We use two convolutional layers with ReLU activation and max pooling to obtain an intermediate representation z ∈ &reals;<sup>H<sub>0</sub>×W<sub>0</sub>×p</sup> so that p is equivalent to the embedding dimension of the transformer backbone used in ViT. We reshape z to z<sub>0</sub> ∈ &reals;<sup>l×p</sup> where l = H<sub>0</sub> × W<sub>0</sub>, which is the length of the embedding sequence. z<sub>0</sub> is then fed to the transformer encoder. The sequential output of the M-layer transformer encoder is followed by a sequence pooling step where the importance weights are assigned for the output sequence, which is subsequently mapped to a similarity score for the lidar-satellite pair in the MLP head. Each batch consists of physically proximal and distant lidar-satellite pairs with N<sub>S</sub> pairs in total, and thus yielding <b>A</b> = [α<sub>1</sub>, α<sub>2</sub>, ...α<sub>N<sub>S</sub></sub> ]<sup>T</sup> ∈ R<sup>N<sub>S</sub></sup>, which we use as similarity scores at inference.

We also built an EBM with a CNN backbone (i.e. ECML-CNN). It takes the same input as the EBM with a CT backbone (i.e. ECML-CT) and also outputs lidar-satellite similarity scores, as well as attempt to predict the satellite image reflecting the location of the vehicle. ECML-CNN consists of 3 down-convolutional layers followed by six ResNet blocks, where the mean is taken along the height, width, and channel dimensions to attain the same network output <b>A</b> as ECML-CT. More details for ECML-CNN are shown in the Appendix of our paper. 

![ECML-CNN](/figures/ECML.png)

Here are predicted satellite images from both models compared to ground truth for KITTI (top), PandaSet (middle), and Custom (bottom) datasets. For each set of 3 images, we show CNN-predicted (left), CT-predicted (middle), and ground truth (right).

![SatPred](/figures/sat_pred.png)

We also compute the SSIM and PSNR to assess the quality of the generated images, which also reflects the accuracy of the localization prediction.  Higher is better.

![ssim_psnr](/figures/ssim_psnr.png)

Here are some examples of online vehicle lidar image queries overlaid on the bottom-left corner of satellite map priors and resulting localizations for KITTI (top), PandaSet (middle), and Custom (bottom) datasets. Black, red, and yellow boxes are ground truth, CT-predicted, and CNN-predicted locations, respectively. The maps are retrieved from a database so that the vehicle location is randomly positioned on the map to make the task more challenging and allow for _one-shot_ localization. </br>

![Results1](/figures/loc1.png)</br>

Sometimes there are multiple structures in a satellite map that appear similar to the online lidar BEV image. A model that has performed better learning would be able to better distinguish the differences and provide accurate similarity scores. Here are some examples of ECML-CT successful predictions but ECML-CNN failures. Note the similarity in appearance between the true locations and the incorrect predictions. Black, red, and yellow boxes are true, CT-predicted, and CNN-predicted locations, respectively. </br>
![Results2](/figures/loc2.png)</br>

We use a 2-stage inference approach that significantly decreases processing time, allowing for real-time inference. In the first stage, we compose a set of input
pairs by skipping m pixels in both x and y coordinates, and n degrees, yielding MS m2 ∗ Nθ n pairs. We then take the top k candidates with the highest similarity scores and perform a 2nd stage of inference where we sweep the area around those candidates in pixel and angle space in 1 pixel and 1◦ increments, respectively. That is, we “fill the gaps” from the 1st stage around the top k candidates. The predicted pose is then derived from the lidar-satellite pair with the highest similarity score of the 2nd stage. We experimented with m ∈ [1, 8] and n ∈ [1, 4] and found inference can be performed in 1.59 sec on a Titan X GPU. The plot below shows pose errors and the corresponding inference time per sample for various m and n values for ECML-CT and ECML-CNN.

![ProcTime](/figures/accuracy_vs_time_combined.png)

Here's a look at the heatmap for ECML-CT (left) and ECML-CNN (middle) along with the map prior (right). One can see that the areas with high similarity scores (darker spots) are usually surrounded by dark regions, indicating that if the optimal location is missed in the first sweep, it will likely be captured in the second sweep.

![Heatmap](/figures/heatmap.png)

We take 100 waypoints from each of our datasets and measure pose prediction accuracy. Below are the results of our two models along with baseline models, which are detailed in the Appendix of our paper. Note that Custom A and Custom B are both derived from our custom dataset collected in two desert towns of southeastern California, but split in two different ways. Custom A is derived from combined data from both towns and then split into training and test, whereas Custom B is derived from splitting the data by town, so that the model is trained on one town and tested on the other town. Custom B is more challenging since both lidar and satellite images are unseen, whereas only lidar is unseen for Custom A. Our model, ECML-CT, achieves superior performance compared to the baselines.

![ResultsTable](/figures/results_table.png)</br></br>

We show the histograms for ECML-CT (green) and ECML-CNN (blue) for all the datasets. ECML-CT is more robust as it has lower variance. </br>
ECML-CT histograms:</br>
![hist_ct1](/figures/hist_kitti_pandaset_xf.png)![hist_ct2](/figures/hist_ntc_ntc_bycity_xf.png)</br>
ECML-CNN histograms:</br>
![hist_cnn1](/figures/hist_kitti_pandaset.png)![hist_cnn2](/figures/hist_ntc_ntc_bycity.png)</br>

Furthermore, we investigate the effect of map coverage area and heading noise have on pose estimation accuracy. See plots below. Although the error increases with increasing map sizes and greater heading uncertainty, ECML-CT has the lowest rate of error increase. We use the KITTI dataset for illustration here.

![mapsize](/figures/accuracy_vs_mapsize_xf.png)</br>
![noise](/figures/accuracy_vs_noise_xf.png)</br>



