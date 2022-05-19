More information regarding this work can be found in [this link](https://ieeexplore.ieee.org/document/9517282).

ResNet50SimCLR.pt contains a pretrained model that can be used directly for inference as shown in `UC4aDemonstration.ipynb`.

More encoders, pretrained in a self-supervised learning fashion, can be found [here](https://www.dropbox.com/s/qcieo92cdyqtjgp/models.zip?dl=0).

  
  #### Example for loading the pre-trained encoders: ###
  
  ```
  backbone = torchvision.models.resnet50(pretrained=False)
  backbone.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), backbone.fc)
  backbone = torch.nn.parallel.DataParallel(backbone,device_ids=[0,1])
  backbone.load_state_dict(torch.load('ResNet50_Simclr_500_Epochs.pt'))
  backbone.module.fc = nn.Identity()
  backbone = backbone.module
  ```
  
  
  `main.py` handles both the self-supervised learning pretraining as well as the supervised training of the linear classifier. 


Examples of the data and the class activation mappings can be found in `UC4aDemonstration.ipynb`.
