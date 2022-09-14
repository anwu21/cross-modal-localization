# Energy-Based Models for Cross-Modal Localization using Convolutional Transformers

If you find this repository useful for your research, please cite our paper:

        @inproceedings{wu2019fisl,
              title={Model-based Behavioral Cloning with Future Image Similarity Learning},
              <--! booktitle={International Conference on Robotics and Automation (ICRA)}, -->
              author={Alan Wu, AJ Piergiovanni, and Michael S. Ryoo},
              year={2019}
        }
        
We present a novel framework using Energy-Based Models (EBMs) for localizing a ground vehicle mounted with a range sensor against satellite imagery in the absence of GPS. Lidar sensors have become ubiquitous on autonomous vehicles for describing its surrounding environment. Map priors are typically built using the same sensor modality for localization purposes. However, these map building endeavors using range sensors are often expensive and time-consuming. Alternatively, we leverage the use of satellite images as map priors, which are widely available, easily accessible, and provide comprehensive coverage. We propose a method using convolutional transformers that performs accurate metric-level localization in a cross-modal manner, which is challenging due to the drastic difference in appearance between the sparse range sensor readings and the rich satellite imagery. We train our model end-to-end and demonstrate our approach achieving higher accuracy than the state-of-the-art on KITTI, Pandaset, and a custom dataset.

![SystemDiagram](/figures/SystemDiagram_smallest.png)

Here is a sample of training videos from a real office environment with various targets:

![kroger](/dataset/office_real/kroger/run1/kroger.gif)![vball](/dataset/office_real/vball/run1/vball.gif)![airfil](/dataset/office_real/airfil/run1/airfil.gif)![tjoes](/dataset/office_real/tjoes/run1/tjoes.gif)

And here is a sample of training videos from a simulated environment (Gazebo) with various obstacles:

![obs1](/dataset/gazebo_sim/obs1/run1/obs1.gif)![obs2](/dataset/gazebo_sim/obs2/run1/obs2.gif)

Sample training data can be found in the folders [/dataset/office_real](/dataset/office_real) and [/dataset/gazebo_sim](/dataset/gazebo_sim). The entire dataset can be downloaded by clicking the link here: <a href="https://iu.box.com/s/nlu8y7yc9863w2yc1pgl9p2s2jxcjlde">Dataset</a>. We use images of size 64x64.

Here is an illustration of the stochastic image predictor model.  This model takes input of the current image and action, but also learns to generate a prior, z<sub>t</sub>, which varies based on the input sequence.  This is further concatenated with the representation before future image prediction. The use of the prior allows for better modeling in stochastic environments and generates clearer images.

![Model](/figures/model_svg.png)

Predicted future images in the real-life lab (top) and simulation (bottom) environments taking different actions. Top two rows of each environment: deterministic model with linear and convolutional state representation, respectively. Bottom two rows: stochastic model with linear and convolutional state representation, respectively. Center image of each row is current image with each adjacent image to the left turning -5° and to the right turning +5°.

![Arc_Lab](/figures/predicted_arc_lab.png)

![Arc Gaz](/figures/predicted_arc_gaz.png)

Sample predicted images from the real and simulation datasets.  From left to right: current image; true next image; deterministic linear; deterministic convolutional; stochastic linear; stochastic convolutional. 

High level description of action taken for each row starting from the top: turn right; move forward; move forward slightly; move forward and turn left; move forward and turn left. 

![Lab](/figures/predicted_lab.png)

High level description of action taken for each row starting from the top: moveforward and turn right; turn right slightly; turn right; move forward slightly; turn left slightly.

![Gaz](/figures/predicted_gaz.png)

Using the stochastic future image predictor, we can generate realistic images to train a critic V_hat that helps select the optimal action:

![Critic](/figures/critic-training.png)

<br />
<br />

We verified our future image prediction model and critic model in real life and in simulation environments. Here are some example trajectories from the real-life robot experiments comparing to baselines (Clone, Handcrafted Critic, and Forward Consistency). Our method is labeled as Critic-FutSim-Stoch. The red ‘X’ marks the location of the target object and the blue ‘∗’ marks the end of each robot trajectory.

![Test trajectories](/figures/imitation_traj_airfil.png)


