# NN-Polar-Decoder

We propose a low-complexity recurrent neural network (RNN) polar decoder with codebook-based weight quantization. Hope this code is useful for peer researchers. If you use this code or parts of it in your research, please kindly cite our paper:

- **Author**: Chieh-Fang Teng, Chen-Hsi (Derek) Wu, Andrew Kuan-Shiuan Ho, and An-Yeu (Andy) Wu
- **Related publications:** Chieh-Fang Teng, Chen-Hsi (Derek) Wu, Andrew Kuan-Shiuan Ho, and An-Yeu (Andy) Wu, "[Low-complexity Recurrent Neural Network-based Polar Decoder with Weight Quantization Mechanism](https://ieeexplore.ieee.org/abstract/document/8683778)," *accepted by 2019 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).*

---

## Required Packages

- python 3.6.5
- numpy 1.16.4
- tensorflow 1.8.0

## Parameters

- Users need to customize the *config.py* and *Polar-NN-MULT.ipynb* as
  - `N` : Block length 
  - `K` : Information length
  - `ebn0` : Desired SNR range 
  - `numOfWord` : Desired batch size 
  - `bp_iter_num` : The number of iteration for BP
  - `RNN` : Whether using recurrent architecture (1 = yes)
  - `quantize_weight` : Different mechanism for weight quantization (0 for non-quantize, 1 for normal, 2 for binarized, 3 for bin, 4 for binarized bin)
  - `bin_bit` : The number of different value
  - `binary_prec` : The number of weight precision (binary_prec must >= bin_bit)

## Contact Information

   ```
Chieh-Fang Teng:
        + jeff@access.ee.ntu.edu.tw
        + d06943020@ntu.edu.tw
   ```
