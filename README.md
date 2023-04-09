# Histology Gene Groups Xplorer(HiGGsXplore)

## Summary
HiGGsXplore is a computational pathology pipleine that provide insight into patient gene expression state form digitized images of H&E stained tissue section.
The framework take a Whole Slide Image (WSI) as input and generate a WSI-Graph. The WSI-Graph is then passed as input to a Graph neural network which predict
the expression state of patient using 200 binary variable (Gene Groups status). Each binary variable represent the collective expression of set of genes genes whose
gene expression patterns are significantly statistically dependent and covarying across breast cancer patients. The Gene Groups and their status are biologically meaningful,
carry histopathological insights and are clinically relevant in terms of association with survival, pharmaco-sensitivity and therapeutic decision-making. 
For details descroption please refer to the preprint: [TODO] 

## Concept Diagram
![image](https://user-images.githubusercontent.com/13537509/230778558-4403a42f-4819-41bf-af2d-53e92f84af05.png)

## Repository Structure

Workspace directory contain necessary script for constructing graph and training the proposed SlideGraph∞
