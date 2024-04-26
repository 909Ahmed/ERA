# YoloV3
________
YoloV3 Simplified for training on Colab with custom dataset. 

### Training log

Namespace(epochs=100, batch_size=16, accumulate=4, cfg='cfg/yolov3-custom.cfg', data='data/customlast/custom.data', multi_scale=False, img_size=[512], rect=False, resume=False, nosave=False, notest=False, evolve=False, bucket='', cache_images=False, weights='weights/yolov3-spp-ultralytics.pt', name='', device='', adam=False, single_cls=False)
Using CUDA device0 _CudaDeviceProperties(name='Tesla T4', total_memory=15102MB)
           device1 _CudaDeviceProperties(name='Tesla T4', total_memory=15102MB)

Model Summary: 225 layers, 6.25841e+07 parameters, 6.25841e+07 gradients
Caching labels (100 found, 0 missing, 0 empty, 0 duplicate, for 100 images): 100
Reading image shapes: 100%|█████████████████| 100/100 [00:00<00:00, 7389.32it/s]
Caching labels (100 found, 0 missing, 0 empty, 0 duplicate, for 100 images): 100
Image sizes 512 - 512 train, 512 test
Using 4 dataloader workers
Starting training for 300 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     0/299     6.48G      4.68      33.3      2.02        40        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100         0         0  0.000364         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     1/299     12.2G      4.28      2.37      1.87      8.52        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100         0         0   0.00192         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     2/299     12.2G      3.25       2.7      1.56      7.51        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100         0         0     0.108         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     3/299     12.2G      3.71      2.65      1.69      8.05        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.303    0.0247     0.292    0.0456

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     4/299     12.2G      3.27      2.11      1.51      6.89        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.137      0.73     0.193     0.229

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     5/299     12.2G       3.5      1.48      1.42       6.4        14       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.145     0.787     0.303     0.243

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     6/299     12.2G      3.88      1.15      1.48      6.51        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.243     0.542     0.274     0.334

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     7/299     12.2G      3.24     0.995      1.28      5.52        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.164     0.392     0.156     0.231

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     8/299     12.2G      3.14      0.97      1.57      5.68        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.254     0.708     0.393     0.373

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     9/299     12.2G      3.16     0.883      1.56      5.61        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.292     0.805     0.446     0.428

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    10/299     12.2G      3.65     0.866       1.3      5.82        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.306     0.735      0.41     0.432

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    11/299     12.2G      3.37     0.889      1.24      5.49        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.32     0.659     0.403     0.431

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    12/299     12.2G      2.99     0.862      1.36      5.21        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100       0.3     0.762     0.376     0.427

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    13/299     12.2G      2.66     0.761     0.989      4.41        14       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100    0.0987     0.498     0.164     0.149

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    14/299     12.2G      3.82     0.843      1.33      5.99        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.156     0.645     0.171     0.225

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    15/299     12.2G      3.31     0.808     0.722      4.84        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.228     0.752      0.28     0.314

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    16/299     12.2G      3.36     0.791     0.709      4.86        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.595     0.776     0.658     0.642

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    17/299     12.2G      3.13     0.731     0.604      4.46        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.445     0.782     0.541     0.561

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    18/299     12.2G      3.01      0.76     0.425       4.2        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.422     0.711     0.466      0.53

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    19/299     12.2G      2.65      0.72     0.388      3.75        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.604     0.868     0.716     0.701

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    20/299     12.2G      2.82      0.78     0.472      4.08        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.584     0.948      0.82     0.723

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    21/299     12.2G      3.01     0.752     0.488      4.24        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.468     0.955     0.715     0.627

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    22/299     12.2G       3.3     0.756     0.509      4.56        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.417     0.758     0.492     0.443

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    23/299     12.2G      2.78     0.797     0.647      4.23        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.451      0.77     0.457     0.489

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    24/299     12.2G      3.89     0.836      0.56      5.29        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.765     0.395      0.64     0.507

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    25/299     12.2G      4.03     0.869     0.606      5.51        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.784     0.754     0.828      0.76

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    26/299     12.2G      2.83     0.712     0.322      3.86        23       512
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.03+/-0.26     -10.81+/-3.68      -0.59+/-0.90 
                         101       0.22+/-0.26     -11.93+/-1.64      -0.52+/-0.95 
                         113       0.22+/-0.54     -10.82+/-1.15      -0.14+/-0.37 
    26/299     12.2G      2.89     0.772     0.388      4.05        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.468     0.523      0.45     0.493

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    27/299     12.2G       3.2     0.766     0.479      4.44        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.804      0.63     0.716     0.701

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    28/299     12.2G      2.81     0.838     0.343      3.99        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.452     0.455     0.398     0.444

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    29/299     12.2G      2.18     0.792     0.268      3.24        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.845     0.675     0.795      0.75

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    30/299     12.2G       2.8     0.769     0.277      3.85        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.795     0.565     0.687     0.628

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    31/299     12.2G      2.63     0.727     0.645      4.01        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.651     0.211     0.262      0.23

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    32/299     12.2G      3.07     0.844     0.641      4.55        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.539      0.33     0.415     0.353

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    33/299     12.2G      2.66     0.828      0.66      4.15        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.173      0.48     0.267     0.212

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    34/299     12.2G      2.31     0.827     0.456      3.59        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.388      0.27     0.223     0.201

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    35/299     12.2G      2.38       0.8     0.495      3.67        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.513     0.537     0.559     0.525

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    36/299     12.2G       2.4     0.795     0.324      3.52        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.773     0.692     0.742      0.73

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    37/299     12.2G      2.12     0.732     0.302      3.16        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.864     0.439     0.655     0.562

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    38/299     12.2G      2.77     0.772     0.437      3.98        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.819     0.659     0.799     0.726

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    39/299     12.2G      2.63     0.729     0.324      3.68        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.831     0.667     0.821      0.74

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    40/299     12.2G      2.11     0.728     0.491      3.33        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.949     0.737     0.898     0.829

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    41/299     12.2G      1.95     0.689     0.259      2.89        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.923      0.38     0.635     0.527

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    42/299     12.2G      2.84     0.705     0.299      3.85        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.78     0.319     0.401     0.435

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    43/299     12.2G      2.57     0.759     0.353      3.69        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.888      0.78     0.873     0.828

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    44/299     12.2G      2.51     0.703     0.224      3.44        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.844     0.865     0.886     0.854

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    45/299     12.2G      2.93     0.717     0.448      4.09        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.813       0.9      0.93     0.854

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    46/299     12.2G      2.19     0.688     0.295      3.17        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.921     0.941     0.956      0.93

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    47/299     12.2G       2.6     0.694     0.441      3.73        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.886     0.858     0.931     0.869

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    48/299     12.2G      2.29      0.69     0.234      3.21        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.77     0.876     0.867     0.809

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    49/299     12.2G      2.11     0.753     0.288      3.15        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.967     0.818      0.95     0.886

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    50/299     12.2G      2.43     0.666     0.327      3.42        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.89     0.426     0.585     0.563

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    51/299     12.2G      2.39       0.7     0.325      3.41        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.818     0.282      0.35     0.382

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    52/299     12.2G      2.22     0.667     0.291      3.17        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.989     0.734     0.933     0.837

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    53/299     12.2G      2.35     0.618     0.195      3.17        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.985     0.769     0.955     0.863

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    54/299     12.2G      2.44     0.667      0.26      3.37        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.942     0.944     0.954     0.943

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    55/299     12.2G       2.2      0.65     0.169      3.02        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.979     0.923     0.985      0.95

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    56/299     12.2G      2.29     0.634      0.15      3.07        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.978     0.935     0.981     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    57/299     12.2G      1.99      0.61     0.117      2.71        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.961     0.979     0.959

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    58/299     12.2G      2.82     0.591     0.411      3.83        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.914     0.945     0.974     0.929

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    59/299     12.2G         2      0.63     0.254      2.88        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.898     0.913     0.959     0.903

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    60/299     12.2G      2.23     0.681     0.178      3.09        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.926     0.972     0.974     0.948

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    61/299     12.2G      2.25     0.634      0.17      3.05        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100       0.9      0.95     0.965     0.924

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    62/299     12.2G      2.02     0.661      0.16      2.84        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.808     0.968     0.949     0.876

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    63/299     12.2G      2.05     0.661     0.157      2.86        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.881     0.982     0.977     0.929

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    64/299     12.2G      2.09     0.574     0.101      2.77        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.93     0.954     0.971     0.941

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    65/299     12.2G      2.02     0.602     0.115      2.74        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.935     0.893     0.966     0.912

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    66/299     12.2G      2.03     0.615     0.117      2.76        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.915     0.964     0.933

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    67/299     12.2G      1.84     0.591     0.175      2.61        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.972     0.941     0.975     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    68/299     12.2G      2.42     0.635     0.133      3.19        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.961     0.953     0.981     0.956

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    69/299     12.2G       1.6     0.612     0.105      2.32        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.939     0.975     0.981     0.956

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    70/299     12.2G      2.11     0.592     0.124      2.83        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.95     0.982     0.983     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    71/299     12.2G      2.65      0.62     0.348      3.61        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.935     0.968     0.978     0.951

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    72/299     12.2G       1.8     0.604     0.227      2.63        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.933      0.97     0.976     0.951

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    73/299     12.2G      1.98     0.611     0.133      2.72        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.925     0.943     0.968     0.934

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    74/299     12.2G      2.39     0.607     0.224      3.22        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.965     0.938      0.98     0.951

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    75/299     12.2G      1.92     0.628     0.154       2.7        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.942     0.938      0.96     0.939

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    76/299     12.2G      1.99     0.619      0.28      2.89        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.923     0.798     0.889     0.856

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    77/299     12.2G      1.79     0.626    0.0923      2.51        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.747     0.714      0.73     0.697

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    78/299     12.2G      1.67     0.583      0.16      2.41        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.938     0.867      0.95     0.901

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    79/299     12.2G      2.11     0.608    0.0992      2.81        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.94     0.924     0.963     0.931

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    80/299     12.2G      2.46     0.598      0.23      3.29        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959     0.968     0.982     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    81/299     12.2G      2.14     0.589    0.0915      2.82        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.978     0.968     0.982     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    82/299     12.2G      1.63     0.643     0.143      2.42        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.929     0.926     0.969     0.927

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    83/299     12.2G      2.56     0.614     0.189      3.36        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.942     0.944     0.976     0.943

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    84/299     12.2G      2.27     0.596    0.0987      2.97        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.94     0.931     0.972     0.934

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    85/299     12.2G      2.35     0.593     0.118      3.06        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.923     0.962     0.982     0.942

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    86/299     12.2G      2.36     0.568     0.121      3.05        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.938     0.955     0.978     0.946

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    87/299     12.2G      1.77     0.602     0.269      2.64        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.92     0.958     0.964     0.939

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    88/299     12.2G      2.12     0.566     0.249      2.93        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.963     0.927     0.974     0.944

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    89/299     12.2G      1.94     0.531      0.18      2.65        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.953     0.968     0.981      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    90/299     12.2G      2.38     0.539     0.168      3.09        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.963     0.977     0.957

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    91/299     12.2G      2.32     0.563     0.177      3.06        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.906     0.982     0.971     0.942

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    92/299     12.2G      1.91     0.566     0.209      2.69        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.886     0.982     0.978     0.929

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    93/299     12.2G      2.68      0.58     0.085      3.35        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.959     0.982     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    94/299     12.2G      2.14     0.555     0.212       2.9        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.965     0.955      0.98      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    95/299     12.2G       1.9     0.609     0.226      2.73        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.963      0.97      0.98     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    96/299     12.2G      2.28     0.582      0.33      3.19        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.944      0.92     0.965     0.932

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    97/299     12.2G      2.29     0.566     0.262      3.12        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.911     0.942     0.952     0.926

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    98/299     12.2G      1.88     0.576     0.173      2.63        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.898     0.937     0.957     0.917

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
    99/299     12.2G         2     0.605     0.222      2.82        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.898     0.978      0.97     0.935

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   100/299     12.2G      2.67     0.546     0.154      3.37        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.935     0.978      0.98     0.956

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   101/299     12.2G      2.05     0.586     0.192      2.83        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.916     0.968     0.974      0.94

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   102/299     12.2G      2.08     0.639     0.216      2.93        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.917      0.93     0.959     0.921

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   103/299     12.2G      1.74     0.608    0.0604      2.41        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.903     0.865     0.915     0.879

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   104/299     12.2G      1.91     0.578     0.142      2.63        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.899     0.857     0.924     0.868

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   105/299     12.2G      1.85     0.563     0.156      2.56        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.943     0.936     0.976     0.937

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   106/299     12.2G      1.62     0.585    0.0593      2.26        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.937     0.954     0.982     0.944

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   107/299     12.2G      1.67     0.561     0.233      2.46        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.918     0.862     0.929     0.885

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   108/299     12.2G      1.62     0.572     0.305       2.5        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.928     0.962     0.979     0.945

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   109/299     12.2G      2.09     0.572     0.121      2.79        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959     0.926     0.978     0.941

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   110/299     12.2G      2.05     0.567     0.159      2.77        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.922     0.932     0.968     0.926

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   111/299     12.2G      1.68     0.595     0.116      2.39        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.912     0.931     0.963      0.92

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   112/299     12.2G      1.81     0.593     0.138      2.54        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.907     0.968     0.979     0.936

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   113/299     12.2G      1.82     0.576     0.173      2.57        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.905     0.955     0.978     0.929

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   114/299     12.2G      1.62     0.543    0.0593      2.22        24       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.937     0.966     0.983     0.951

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   115/299     12.2G      1.98     0.557     0.179      2.72        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.95     0.971     0.984      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   116/299     12.2G      1.53     0.546     0.138      2.21        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.945      0.95     0.979     0.946

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   117/299     12.2G      2.36     0.538     0.149      3.05        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.938     0.977     0.947

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   118/299     12.2G      1.41      0.53    0.0755      2.01        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.96     0.918     0.973     0.939

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   119/299     12.2G      1.55     0.508     0.107      2.17        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.939      0.95     0.975     0.944

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   120/299     12.2G      1.86      0.51     0.124      2.49        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946     0.962      0.98     0.954

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   121/299     12.2G       2.1     0.511    0.0542      2.67        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.963     0.979     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   122/299     12.2G       1.4     0.592    0.0848      2.07        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.964     0.966     0.984     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   123/299     12.2G      1.34     0.506    0.0575       1.9        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.964     0.974     0.985     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   124/299     12.2G      1.39     0.504    0.0642      1.96        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.942     0.978     0.984     0.959

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   125/299     12.2G       2.3     0.522    0.0597      2.88        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.941     0.971     0.982     0.956

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   126/299     12.2G      1.95      0.52    0.0735      2.55        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957      0.98     0.985     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   127/299     12.2G      1.33     0.479    0.0428      1.85        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.974      0.98     0.988     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   128/299     12.2G      2.34     0.474    0.0586      2.87        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.974     0.988     0.989     0.981

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   129/299     12.2G      1.61     0.498    0.0347      2.14        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.983     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   130/299     12.2G       1.7     0.471    0.0659      2.24        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.956     0.984     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   131/299     12.2G      1.66      0.48    0.0526      2.19        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.96      0.99     0.986     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   132/299     12.2G       1.3     0.488    0.0341      1.82        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.949      0.99     0.985     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   133/299     12.2G       1.5     0.442    0.0406      1.98        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.96     0.967     0.984     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   134/299     12.2G      1.98     0.455     0.137      2.57        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.969     0.965     0.986     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   135/299     12.2G      1.83     0.489    0.0463      2.37        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.978     0.987     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   136/299     12.2G      1.87     0.453    0.0918      2.42        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.962     0.988     0.987     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   137/299     12.2G      1.87     0.499      0.12      2.49        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.945     0.983     0.988     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   138/299     12.2G       2.6     0.457      0.08      3.14        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.937     0.975     0.988     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   139/299     12.2G      1.45      0.48     0.157      2.09        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.938     0.983     0.987      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   140/299     12.2G      1.45     0.483    0.0415      1.97        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.986     0.988     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   141/299     12.2G      1.37     0.488    0.0702      1.93        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.991     0.988     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   142/299     12.2G      2.06     0.484    0.0984      2.65        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.964     0.985     0.988     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   143/299     12.2G      1.79     0.477    0.0399       2.3        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.971     0.982     0.988     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   144/299     12.2G      1.83     0.467    0.0568      2.35        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.98     0.978     0.989     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   145/299     12.2G      1.79     0.503    0.0495      2.34        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.956     0.985     0.988      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   146/299     12.2G      1.77     0.444    0.0918      2.31        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.96     0.989     0.987     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   147/299     12.2G      1.63     0.449    0.0422      2.12        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.974     0.982     0.987     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   148/299     12.2G      2.31     0.462     0.041      2.81        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.979     0.978     0.987     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   149/299     12.2G       1.6      0.45    0.0906      2.14        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957      0.98     0.987     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   150/299     12.2G      1.57     0.472     0.053       2.1        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.956     0.982     0.987     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   151/299     12.2G      1.59     0.465    0.0561      2.11        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.982     0.986      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   152/299     12.2G      1.37     0.455    0.0862      1.91        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.958     0.983     0.986      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   153/299     12.2G      1.73     0.434    0.0537      2.22        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.967     0.985     0.987     0.976

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   154/299     12.2G      1.55      0.47    0.0508      2.07        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957      0.99     0.989     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   155/299     12.2G      2.01     0.439    0.0237      2.47        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.961      0.99     0.989     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   156/299     12.2G      1.42      0.45    0.0769      1.95        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.961      0.99      0.99     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   157/299     12.2G      2.14     0.424    0.0658      2.63        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.966      0.99     0.989     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   158/299     12.2G      1.33     0.434     0.223      1.99        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.949     0.988     0.988     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   159/299     12.2G      1.32     0.446      0.13       1.9        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.943     0.988     0.987     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   160/299     12.2G      1.55     0.477     0.163      2.19        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.937     0.982     0.986     0.959

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   161/299     12.2G      1.55     0.466     0.264      2.28        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.912     0.988     0.983     0.948

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   162/299     12.2G      1.93     0.482     0.108      2.52        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.923     0.988     0.983     0.954

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   163/299     12.2G      2.11     0.487      0.11      2.71        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.924     0.984     0.985     0.953

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   164/299     12.2G       1.5     0.458    0.0525      2.01        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.932     0.982     0.986     0.956

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   165/299     12.2G      1.77     0.456     0.113      2.34        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.975     0.986     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   166/299     12.2G      1.52     0.527      0.27      2.32        24       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.965     0.992      0.99     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   167/299     12.2G      2.06     0.486     0.081      2.63        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.978      0.98     0.988     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   168/299     12.2G      2.32     0.474    0.0624      2.86        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.963     0.973     0.982     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   169/299     12.2G      1.59     0.463    0.0837      2.14        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.962     0.978     0.988      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   170/299     12.2G      1.27     0.485    0.0577      1.81        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.942     0.981     0.987     0.961

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   171/299     12.2G       2.1     0.472      0.07      2.64        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.934     0.983     0.987     0.958

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   172/299     12.2G      1.32     0.437     0.073      1.83        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946     0.985     0.987     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   173/299     12.2G       2.1     0.445    0.0612      2.61        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.985     0.987     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   174/299     12.2G      1.44     0.419    0.0302      1.89        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.956     0.985     0.987      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   175/299     12.2G      1.45     0.439     0.056      1.95        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.963     0.985     0.987     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   176/299     12.2G       1.5     0.479    0.0269      2.01        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.965     0.985     0.987     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   177/299     12.2G      1.42     0.426    0.0331      1.88        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957      0.99     0.986     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   178/299     12.2G       1.3     0.394    0.0502      1.74        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954      0.99     0.987     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   179/299     12.2G      1.85     0.418    0.0374      2.31        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.96     0.985     0.988     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   180/299     12.2G      1.77     0.385    0.0692      2.22        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.985     0.988      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   181/299     12.2G      1.46      0.41     0.027       1.9        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.947     0.988     0.988     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   182/299     12.2G      1.21      0.41    0.0357      1.65        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.987     0.988     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   183/299     12.2G      1.04     0.434    0.0406      1.52        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.947     0.987     0.988     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   184/299     12.2G      1.61     0.401     0.035      2.05        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955     0.985     0.987     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   185/299     12.2G      1.45     0.389    0.0285      1.87        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.985     0.987     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   186/299     12.2G      1.75     0.381    0.0211      2.16        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.943      0.99     0.987     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   187/299     12.2G      1.39      0.38    0.0524      1.82        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.942      0.99     0.987     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   188/299     12.2G      1.58     0.385    0.0247      1.99        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.941      0.99     0.986     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   189/299     12.2G      1.57     0.387    0.0409         2        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.95      0.99     0.988     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   190/299     12.2G      1.42     0.404    0.0244      1.85        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.949      0.99     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   191/299     12.2G      1.78     0.351    0.0165      2.15        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.943      0.99     0.989     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   192/299     12.2G      1.52     0.387    0.0204      1.93        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948      0.99     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   193/299     12.2G      2.22     0.361    0.0374      2.62        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.94      0.99     0.989     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   194/299     12.2G      1.33     0.402    0.0218      1.75        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948      0.99     0.988     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   195/299     12.2G      1.51     0.381     0.372      2.27        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.95      0.99     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   196/299     12.2G      1.35     0.402    0.0571      1.81        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.988     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   197/299     12.2G      1.11     0.371    0.0695      1.55        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955     0.988     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   198/299     12.2G      1.72     0.373     0.148      2.24        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.944     0.988     0.989     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   199/299     12.2G      1.33     0.388     0.027      1.74        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.941     0.982     0.989     0.961

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   200/299     12.2G      1.31     0.366    0.0415      1.72        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959     0.988     0.989     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   201/299     12.2G      1.14     0.358    0.0251      1.53        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959     0.986     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   202/299     12.2G      1.16     0.393    0.0229      1.58        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.949     0.985     0.989     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   203/299     12.2G      2.08     0.341    0.0226      2.45        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946     0.985     0.989     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   204/299     12.2G      1.95     0.355    0.0296      2.33        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.939     0.983     0.987      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   205/299     12.2G      1.14     0.404    0.0359      1.58        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.935     0.981     0.987     0.958

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   206/299     12.2G      2.02     0.377    0.0183      2.41        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.937      0.98     0.986     0.958

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   207/299     12.2G      1.66     0.364    0.0216      2.05        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.941      0.98     0.986      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   208/299     12.2G      2.06      0.35    0.0248      2.43        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946     0.983     0.986     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   209/299     12.2G      1.28      0.37    0.0315      1.68        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.945     0.985     0.987     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   210/299     12.2G      1.24     0.351    0.0319      1.62        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951     0.985     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   211/299     12.2G      1.63     0.342    0.0199      1.99        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.985     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   212/299     12.2G      1.07     0.352    0.0451      1.47        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955     0.985     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   213/299     12.2G      1.25     0.358    0.0274      1.64        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.985     0.989     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   214/299     12.2G      1.83      0.35    0.0169       2.2        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.944     0.985     0.989     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   215/299     12.2G      1.62     0.343    0.0189      1.98        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   216/299     12.2G      1.62     0.339    0.0201      1.97        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957     0.985     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   217/299     12.2G      1.26     0.344    0.0496      1.65        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   218/299     12.2G      1.22      0.33    0.0149      1.56        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.937      0.99     0.988     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   219/299     12.2G      1.26     0.329    0.0276      1.62        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.941     0.989     0.988     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   220/299     12.2G      1.63      0.34    0.0213      1.99        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.947     0.988     0.989     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   221/299     12.2G      1.01     0.355    0.0208      1.39        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.939     0.988     0.988     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   222/299     12.2G      1.19     0.303    0.0138      1.51        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946     0.983     0.988     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   223/299     12.2G      1.59     0.313    0.0149      1.92        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.95     0.982     0.988     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   224/299     12.2G       1.4     0.319    0.0471      1.76        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946     0.984     0.989     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   225/299     12.2G      2.55     0.325    0.0411      2.92        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946     0.985     0.989     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   226/299     12.2G      0.66     0.331    0.0128         1        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.985     0.989     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   227/299     12.2G      1.42     0.306    0.0152      1.74        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951     0.985     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   228/299     12.2G      1.21     0.325    0.0125      1.54        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.961     0.985     0.988     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   229/299     12.2G      1.19      0.35    0.0229      1.56        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.962     0.985     0.988     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   230/299     12.2G      1.36     0.347    0.0152      1.72        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.985     0.988     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   231/299     12.2G      1.02      0.31    0.0103      1.34        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   232/299     12.2G      1.35     0.329    0.0121      1.69        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955     0.985     0.988      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   233/299     12.2G      1.56     0.315    0.0177      1.89        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955     0.985     0.988      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   234/299     12.2G      1.56     0.293    0.0129      1.87        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.96     0.985     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   235/299     12.2G      1.35     0.279    0.0119      1.64        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.985     0.989     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   236/299     12.2G      1.54     0.331    0.0114      1.88        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.95     0.987     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   237/299     12.2G     0.949      0.31    0.0106      1.27        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.982     0.989     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   238/299     12.2G      1.13     0.326     0.014      1.48        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.948     0.982     0.989     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   239/299     12.2G     0.922     0.309   0.00923      1.24        20       512

   242/299     12.2G      1.13     0.292   0.00992      1.43        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   243/299     12.2G     0.721     0.294   0.00982      1.02        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.953     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   244/299     12.2G      1.55     0.276    0.0109      1.84        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.985     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   245/299     12.2G      1.13     0.297    0.0369      1.46        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.956     0.985     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   246/299     12.2G      1.35     0.308     0.012      1.67        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.964     0.985     0.989     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   247/299     12.2G      1.28     0.301    0.0143       1.6        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.965     0.985     0.989     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   248/299     12.2G      1.12      0.31      0.01      1.44        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.968     0.988     0.989     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   249/299     12.2G      1.48     0.296    0.0118      1.79        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.97     0.987     0.989     0.978

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   250/299     12.2G     0.907     0.294    0.0101      1.21        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.971     0.988     0.989     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   251/299     12.2G      1.47     0.306    0.0111      1.79        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.97     0.988     0.989     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   252/299     12.2G      1.09     0.283    0.0126      1.39        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.965     0.985     0.989     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   253/299     12.2G      1.69     0.305    0.0179      2.01        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.964      0.99     0.989     0.977

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   254/299     12.2G     0.891     0.288   0.00929      1.19        14       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957      0.99     0.989     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   255/299     12.2G      1.28     0.294     0.025       1.6        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955      0.99     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   256/299     12.2G      1.12      0.28   0.00918      1.41        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955      0.99     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   257/299     12.2G     0.885       0.3   0.00905      1.19        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959      0.99     0.989     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   258/299     12.2G      1.06     0.257    0.0102      1.33        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959      0.99     0.989     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   259/299     12.2G      1.49     0.311    0.0923      1.89        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959      0.99     0.989     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   260/299     12.2G       1.5      0.28    0.0131      1.79        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.96      0.99      0.99     0.975

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   261/299     12.2G      1.07     0.278    0.0437      1.39        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952      0.99      0.99     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   262/299     12.2G     0.729     0.304     0.019      1.05        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951     0.986     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   263/299     12.2G      1.06     0.274   0.00883      1.35        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.987     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   264/299     12.2G      1.48     0.279    0.0105      1.77        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955     0.988      0.99     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   265/299     12.2G     0.892     0.319    0.0132      1.22        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.988      0.99      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   266/299     12.2G      1.11     0.301    0.0121      1.43        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.949     0.988      0.99     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   267/299     12.2G      1.29       0.3    0.0108       1.6        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.985      0.99     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   268/299     12.2G      1.28     0.306     0.044      1.63        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   269/299     12.2G      1.07     0.284   0.00903      1.36        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   270/299     12.2G      1.46     0.267    0.0115      1.74        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.953     0.985     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   271/299     12.2G      1.08     0.301    0.0107      1.39        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952     0.985     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   272/299     12.2G      1.82     0.301    0.0184      2.14        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951     0.987     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   273/299     12.2G      1.05     0.269   0.00962      1.33        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.958      0.99      0.99     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   274/299     12.2G      1.06     0.299   0.00964      1.37        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.958      0.99      0.99     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   275/299     12.2G      1.82     0.293    0.0101      2.12        20       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959      0.99      0.99     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   276/299     12.2G     0.874     0.257    0.0083      1.14        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952      0.99      0.99     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   277/299     12.2G      1.59     0.269   0.00861      1.86        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.947      0.99      0.99     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   278/299     12.2G      0.68     0.261   0.00839      0.95        22       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.947      0.99      0.99     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   279/299     12.2G      1.04     0.265   0.00808      1.31        23       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.947      0.99      0.99     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   280/299     12.2G      1.25     0.267   0.00893      1.53        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.957      0.99      0.99     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   281/299     12.2G      1.44     0.284   0.00843      1.73        14       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.956      0.99      0.99     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   282/299     12.2G     0.835     0.269   0.00678      1.11        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951      0.99     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   283/299     12.2G      1.79     0.253    0.0117      2.06        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951      0.99     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   284/299     12.2G      1.24      0.28   0.00841      1.53        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955      0.99     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   285/299     12.2G      1.45     0.251   0.00719      1.71        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.955      0.99     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   286/299     12.2G      1.22     0.278   0.00742      1.51        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.949      0.99     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   287/299     12.2G     0.868     0.275   0.00873      1.15        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.946      0.99     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   288/299     12.2G     0.664      0.25   0.00712     0.921        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.947      0.99     0.989     0.968

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   289/299     12.2G       1.6     0.279     0.011      1.89        18       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951      0.99     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   290/299     12.2G      1.21     0.277   0.00748       1.5        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952      0.99     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   291/299     12.2G      1.04      0.25   0.00763       1.3        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954      0.99     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   292/299     12.2G      1.04     0.263    0.0075      1.32        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.953     0.986     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   293/299     12.2G      1.02     0.247    0.0213      1.28        15       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959     0.989     0.989     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   294/299     12.2G      1.05     0.267    0.0067      1.32        19       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.954      0.99     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   295/299     12.2G     0.838       0.3   0.00663      1.14        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.952      0.99     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   296/299     12.2G      1.61      0.27     0.134      2.01        24       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100      0.95      0.99     0.989     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   297/299     12.2G     0.996     0.278   0.00999      1.28        16       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951      0.99     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   298/299     12.2G      1.23     0.283   0.00895      1.52        17       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.951      0.99     0.989      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
   299/299     12.2G      1.06     0.309    0.0383       1.4        21       512

               Class    Images   Targets         P         R   mAP@0.5        F1
                 all       100       100     0.959      0.99     0.989     0.974
300 epochs completed in 2.229 hours.



**Results**

I used genshin Impact character dataset for the training. I haven't played the game and ended up mislabling the characters 😅. The names are actually Hu Tao, Albedo and Ayaka

![image](https://github.com/909Ahmed/ERA/blob/master/S12/output.png)