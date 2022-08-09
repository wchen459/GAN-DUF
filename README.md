# A Hierarchical Deep Generative Model for Design Under Free-Form Geometric Uncertainty

Experiment code associated with our _IDETC 2022_ paper: [GAN-DUF: Hierarchical Deep Generative Models for Design Under Free-Form Geometric Uncertainty](https://arxiv.org/pdf/2112.08919.pdf).

GAN-DUF is short for __Generative Adversarial Network-based Design under Uncertainty Framework__.

![Alt text](/architecture.svg)

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

   Chen, W. W., Lee, D., & Chen, W. (2022, August). Hierarchical Deep Generative Models for Design Under Free-Form Geometric Uncertainty. In _ASME 2022 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC-CIE)_. American Society of Mechanical Engineers (ASME). (Accepted)

	@inproceedings{chen2022ganduf,
        title={Hierarchical Deep Generative Models for Design Under Free-Form Geometric Uncertainty},
        author={Chen, Wayne Wei and Lee, Doksoo and Chen, Wei},
        booktitle={International Design Engineering Technical Conferences and Computers and Information in Engineering Conference},
        year={2022},
        organization={American Society of Mechanical Engineers}
      }

## Usage

### Obtain dataset

1. Download data (NPY files) from [here](https://drive.google.com/drive/folders/1Q0DKS_kZleIL8GHwppCHyIxmym9l8td9?usp=sharing), and put them in corresponding data directories (`metasurface/data/` or `airfoil/data/`).


### Create virtual environment

1. Go to the code directory. Create the environment from the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```
   
2. Activate the new environment: 

   ```bash
   conda activate ganduf
   ```

### Train generative model

1. Go to example directory (`metasurface` or `airfoil`).

2. Train model:

   ```bash
   python main.py train
   ```

   The values of the model and training configuration will be read from the file `config.ini`.

   The trained model and the result plots will be saved under the directory `trained_model/<parent_latent_dim>_<child_latent_dim>/`, where `<parent_latent_dim>` and `<child_latent_dim>` are parent and child latent dimensions, respectively, and are specified in `config.ini`. 
      

### Test the trained model

1. Generate result plots using the trained model:

   ```bash
   python main.py test
   ```
