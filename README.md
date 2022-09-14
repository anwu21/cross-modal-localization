# Energy-Based Models for Cross-Modal Localization using Convolutional Transformers

If you find this repository useful for your research, please cite our paper:

        @inproceedings{wu2019fisl,
              title={Model-based Behavioral Cloning with Future Image Similarity Learning},
              author={Alan Wu and Michael S. Ryoo},
              year={2022}
        }
        
We present a novel framework using Energy-Based Models (EBMs) for localizing a ground vehicle mounted with a range sensor against satellite imagery in the absence of GPS. Lidar sensors have become ubiquitous on autonomous vehicles for describing its surrounding environment. Map priors are typically built using the same sensor modality for localization purposes. However, these map building endeavors using range sensors are often expensive and time-consuming. Alternatively, we leverage the use of satellite images as map priors, which are widely available, easily accessible, and provide comprehensive coverage. We propose a method using convolutional transformers that performs accurate metric-level localization in a cross-modal manner, which is challenging due to the drastic difference in appearance between the sparse range sensor readings and the rich satellite imagery. We train our model end-to-end and demonstrate our approach achieving higher accuracy than the state-of-the-art on KITTI, Pandaset, and a custom dataset.

![SystemDiagram](/figures/SystemDiagram_smallest.png)

![SystemArch](/figures/CLECT.png)

![SatPred](/figures/sat_pred.png)
![ssim_psnr](/figures/ssim_psnr.png)

![Results1](/figures/loc1.png)
![Results2](/figures/loc2.png)
![ResultsTable](/figures/results_table.png)
![hist_ct](/figures/histogram_ct.png)![hist_cnn](/figures/histogram_cnn.png)




Here is a sample of training videos from a real office environment with various targets:

![kroger](/dataset/office_real/kroger/run1/kroger.gif)![vball](/dataset/office_real/vball/run1/vball.gif)![airfil](/dataset/office_real/airfil/run1/airfil.gif)![tjoes](/dataset/office_real/tjoes/run1/tjoes.gif)

And here is a sample of training videos from a simulated environment (Gazebo) with various obstacles:

![obs1](/dataset/gazebo_sim/obs1/run1/obs1.gif)![obs2](/dataset/gazebo_sim/obs2/run1/obs2.gif)

Sample training data can be found in the folders [/dataset/office_real](/dataset/office_real) and [/dataset/gazebo_sim](/dataset/gazebo_sim). The entire dataset can be downloaded by clicking the link here: <a href="https://iu.box.com/s/nlu8y7yc9863w2yc1pgl9p2s2jxcjlde">Dataset</a>. We use images of size 64x64.

Here is an illustration of the stochastic image predictor model.  This model takes input of the current image and action, but also learns to generate a prior, z<sub>t</sub>, which varies based on the input sequence.  This is further concatenated with the representation before future image prediction. The use of the prior allows for better modeling in stochastic environments and generates clearer images.

![Model](/figures/model_svg.png)

Predicted future images in the real-life lab (top) and simulation (bottom) environments taking different actions. Top two rows of each environment: deterministic model with linear and convolutional state representation, respectively. Bottom two rows: stochastic model with linear and convolutional state representation, respectively. Center image of each row is current image with each adjacent image to the left turning -5° and to the right turning +5°.




