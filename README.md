# Histology Gene Groups Xplorer(HiGGsXplore) 

## Summary
HiGGsXplore is a computational pathology pipleine that provide insight into patient gene expression state form digitized images of H&E stained tissue section.
The framework take a Whole Slide Image (WSI) as input and generate a WSI-Graph. The WSI-Graph is then passed as input to a Graph neural network which predict
the expression state of patient using 200 binary variable (Gene Groups status). Each binary variable represent the collective expression of set of genes genes whose
gene expression patterns are significantly statistically dependent and covarying across breast cancer patients. The Gene Groups and their status are biologically meaningful,
carry histopathological insights and are clinically relevant in terms of association with survival, pharmaco-sensitivity and therapeutic decision-making. 
For details descroption please refer to the preprint: [TODO] 

## Live Demo (<a href='https://tiademos.dcs.warwick.ac.uk/bokeh_app?demo=HiGGsXplore'>Webserver</a>) 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/13537509/230781325-477a60ac-2229-46b5-96f3-6892c6eaf7d6.gif)


## Concept Diagram
![image](https://user-images.githubusercontent.com/13537509/230778558-4403a42f-4819-41bf-af2d-53e92f84af05.png)

## Training and Evaluation

Workspace directory contain necessary script for constructing graph and training the proposed SlideGraph<sup>∞</sup>. 

Step1: Download TCGA BRCA Diagnostic slides from <a href='https://docs.gdc.cancer.gov/Data_Portal/Users_Guide/Repository/'>GCD data portal</a>

Step2: Download tissue segmentation mask from this <a href = "https://drive.google.com/file/d/1nvGyMm33gl-iYlVEziM_RjpL1c61ApXv/view?usp=sharing"> Link</a>.

Step3: Generate patches of each Whole slide image by running
  ```python patches_extraction.py```

Step4: Extract ShuffleNet representation from each of the WSI patch by running
   ```python deep_features.py```

Step5: Construct WSI-Graph by running
   ```python graph_construction.py```

Step6: Training the Graph Neural Network by running
   ```python main.py```

## Running Inference on CPTAC Cohort
Step1: Download WSIs of patients in CPTAC-BRCA cohort from <a href = "[https://drive.google.com/file/d/1nvGyMm33gl-iYlVEziM_RjpL1c61ApXv/view?usp=sharing](https://pathdb.cancerimagingarchive.net/imagesearch?f[0]=collection:cptac_brca)"> CANCR IMAGING ARCHIVE</a>.

Step2: Use the same patch-extraction and feature-extraction pipeline and construct graph representation.

Step3: Run the Inference. 
 ```python inference.py```

## Gene Group 3 Dashboard
  ![G3_github](https://user-images.githubusercontent.com/13537509/230782124-521dcd4e-89f1-4adc-9d18-60683353a387.png)

  
## License
The source code of SlideGraph<sup>∞</sup> is released under MIT license.

## Cite this repo

<pre> @article{
  dawood2023cross,
  title={Cross-linking breast tumor transcriptomic states and tissue histology},
  author={Dawood, Muhammad and Eastwood, Mark and Jahanifar, Mostafa and Young, Lawrence and Ben-Hur, Asa and Branson, Kim and Jones, Louise and Rajpoot, Nasir and Fayyaz, Minhas},
  journal={Cell Reports Medicine},
  volume={4},
  number={12},
  year={2023},
  publisher={Elsevier}
  } </pre>

