# Learning Object Detectors with Semi-Annotated Weak Labels(ODSAWL)


## Installation MatConvNet and WSDDN
1. Download and install [MatConvNet](http://www.vlfeat.org/matconvnet/install/)
2. Install this module with the package manager of MatConvNet [`vl_contrib`](http://www.vlfeat.org/matconvnet/mfiles/vl_contrib/#notes):

```
    vl_contrib('install', 'WSDDN') ;
    vl_contrib('setup', 'WSDDN') ;
```

3. If you want to train a ODSAWL model, download the items below:

    a.  [PASCAL VOC 2007 devkit and dataset](http://host.robots.ox.ac.uk/pascal/VOC/) under `data` folder

    b.  Pre-computed edge-boxes for selectiveSearch-boxes for PASCAL VOC 2007 from [GoolgeDrive](https://drive.google.com/drive/folders/1WXxErFMjZ013xpSfjQWJMVZYt5dpYbNB?usp=sharing)
    
    c. Pre-trained network from [MatConvNet website](http://www.vlfeat.org/matconvnet/models) under `model` folder

## Train and Test

After completing the installation and downloading the required files, you can train and test ODSAWL:

```matlab
            cd scripts;
            opts.modelPath = '....' ;
            opts.gpu = .... ;
	    opts.labelNumPerCls = ...;
	    opts.iteNum = ...;
            odsawl(opts) ;
                        
```
